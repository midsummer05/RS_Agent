from rs_agent.evaluation import QAHarness


def test_qa_harness_reports_recall_and_precision_for_fixed_labelled_cases(tmp_path):
    report = QAHarness.run("data/qa_cases")
    assert report["summary"]["recall"] == 1.0
    assert report["summary"]["precision"] == 1.0
    paths = QAHarness.write(report, tmp_path)
    assert all(path.is_file() for path in paths.values())
