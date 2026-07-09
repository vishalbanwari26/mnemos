import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384

TZDateTime = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())

    episodic_memories: Mapped[list["EpisodicMemory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    semantic_memories: Mapped[list["SemanticMemory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class EpisodicMemory(Base):
    """A single conversational turn, timestamped with both real insert time
    (`created_at`) and logical event time (`occurred_at`). The two are decoupled
    so benchmarks can simulate memories aging over weeks without waiting weeks.
    """

    __tablename__ = "episodic_memories"
    __table_args__ = (
        Index("ix_episodic_user_occurred", "user_id", "occurred_at"),
        Index(
            "ix_episodic_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[uuid.UUID] = mapped_column(index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    memory_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    occurred_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow)

    user: Mapped["User"] = relationship(back_populates="episodic_memories")


class SemanticMemory(Base):
    """A distilled fact about a user, extracted from one or more episodes."""

    __tablename__ = "semantic_memories"
    __table_args__ = (
        Index("ix_semantic_user_status", "user_id", "status"),
        Index(
            "ix_semantic_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    fact: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    source_episode_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    confidence: Mapped[float] = mapped_column(default=1.0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    memory_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, server_default=func.now(), onupdate=utcnow
    )
    # Bumped whenever this fact is actually retrieved and used in a reply.
    # Reflection's decay pass uses staleness since this timestamp (not
    # created_at) to implement "reinforcement through reuse."
    last_reinforced_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="semantic_memories")


class ReflectionLogEntry(Base):
    """Audit trail for reflection actions (merge/decay/forget). Always in
    Postgres regardless of STORAGE_BACKEND — operational metadata, not the
    memory content the backend comparison is about.
    """

    __tablename__ = "reflection_log"
    __table_args__ = (Index("ix_reflection_log_user", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(16))  # "merge" | "decay" | "forget"
    fact_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())


class ProceduralStrategy(Base):
    """Per-user, per-strategy retrieval outcome counters. Always in Postgres
    regardless of STORAGE_BACKEND.
    """

    __tablename__ = "procedural_strategies"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    strategy_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    uses: Mapped[int] = mapped_column(default=0)
    successes: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, server_default=func.now(), onupdate=utcnow
    )


class ProceduralTurn(Base):
    """One row per active session: the strategy chosen for the assistant's
    most recent reply, awaiting an outcome assessment. Resolved (and deleted)
    at the start of the *next* turn in the same session, once the user's
    following message is available to check for a correction cue.
    """

    __tablename__ = "procedural_turns"

    session_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    strategy_name: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
