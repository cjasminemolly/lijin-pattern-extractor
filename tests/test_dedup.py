"""Tests for deduplication module"""
import pytest
from pathlib import Path
from src.extractor.dedup import LijinDeduplicator


def test_deduplicator_init():
    d = LijinDeduplicator()
    assert d.hash_threshold == 8
    assert d.similarity_threshold == 0.92


def test_find_duplicates_empty_dir(tmp_path):
    d = LijinDeduplicator()
    result = d.find_duplicates(str(tmp_path))
    assert result == []


def test_find_duplicates_single_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(tmp_path / "test.jpg")
    d = LijinDeduplicator()
    result = d.find_duplicates(str(tmp_path))
    assert result == []


def test_find_duplicates_identical_images(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(tmp_path / "a.jpg")
    img.save(tmp_path / "b.jpg")
    d = LijinDeduplicator(hash_threshold=0)
    groups = d.find_duplicates(str(tmp_path))
    assert len(groups) >= 1
