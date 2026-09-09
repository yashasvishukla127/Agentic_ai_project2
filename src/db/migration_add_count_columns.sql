-- Migration to add chunks_skipped and chunks_added columns to ingestion_runs table
-- Run this script on existing databases to add the new tracking columns

ALTER TABLE ingestion_runs 
ADD COLUMN IF NOT EXISTS chunks_skipped INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS chunks_added INTEGER NOT NULL DEFAULT 0;
