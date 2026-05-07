import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inference_id = Column(String, unique=True, index=True, nullable=False)
    model_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    features = Column(JSONB, nullable=False)
    prediction = Column(Float, nullable=False)
    confidence = Column(Float)
    ground_truth = Column(Float)
    latency_ms = Column(Float)
    metadata_ = Column("metadata", JSONB, default={})


class DriftResult(Base):
    __tablename__ = "drift_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, index=True, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    feature_name = Column(String, nullable=False)
    test_type = Column(String, nullable=False)  # ks, psi, chi2, kl
    statistic = Column(Float, nullable=False)
    p_value = Column(Float)
    drift_detected = Column(Boolean, nullable=False)
    severity = Column(String, nullable=False)  # INFO, WARNING, CRITICAL


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    severity = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB, default={})
    explanation = Column(Text)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
