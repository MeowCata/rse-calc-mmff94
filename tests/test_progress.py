"""Fast tests for progress feedback and elapsed-time reporting."""

import json

import pytest

from cli.main import main
from ring_strain.core import ProgressUpdate, StrainAnalyzer


def test_progress_update_percent_is_overall_progress():
    update = ProgressUpdate(
        stage="sampling",
        task="Searching torsions",
        progress=0.375,
        elapsed_seconds=1.0,
        smiles="C1CC1",
        current=3,
        total=8,
    )
    assert update.percent == pytest.approx(37.5)


def test_unsupported_analysis_reports_monotonic_progress_and_elapsed():
    events = []
    report = StrainAnalyzer().analyze("c1ccccc1", progress_callback=events.append)

    assert [event.stage for event in events][0] == "start"
    assert [event.stage for event in events][-1] == "complete"
    assert [event.progress for event in events] == sorted(
        event.progress for event in events
    )
    assert events[-1].progress == pytest.approx(1.0)
    assert report.elapsed_time_seconds >= 0.0
    assert "Total elapsed time:" in report.print_summary()
    assert "elapsed_time_seconds" in report.to_dict()


def test_progress_callback_failure_does_not_abort_analysis():
    def broken_callback(_update):
        raise RuntimeError("display failed")

    report = StrainAnalyzer().analyze(
        "c1ccccc1", progress_callback=broken_callback
    )
    assert not report.is_supported


def test_invalid_smiles_emits_error_without_false_completion():
    events = []
    with pytest.raises(ValueError):
        StrainAnalyzer().analyze("not_a_smiles!!", progress_callback=events.append)

    assert events[-1].stage == "error"
    assert events[-1].progress < 1.0
    assert all(event.stage != "complete" for event in events)


def test_cli_json_keeps_progress_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mmff94-strain", "c1ccccc1", "--json"])
    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["is_supported"] is False
    assert "Starting analysis" in captured.err
    assert "Analysis complete" in captured.err


def test_cli_no_progress_is_quiet(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["mmff94-strain", "c1ccccc1", "--json", "--no-progress"]
    )
    main()

    captured = capsys.readouterr()
    json.loads(captured.out)
    assert captured.err == ""
