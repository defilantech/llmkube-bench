from runner.cluster import FakeCluster, KubectlCluster


def test_fake_cluster_records_serve_and_returns_record():
    fc = FakeCluster(
        terminal_status={"verdict": "GO", "result": {"extra": {"branch": "b"}}},
        transcript={"turnCount": 1, "messages": []},
    )
    fc.serve_only("ornith-35b", isvc_all=["ornith-35b", "strix-coder"])
    assert fc.active == "ornith-35b"
    task = fc.dispatch(agent="a", repo="r", issue=433, branch="foreman/bench-x")
    status = fc.wait_terminal(task)
    assert status["verdict"] == "GO"
    assert fc.transcript_of(task)["turnCount"] == 1


def test_kubectl_serve_only_builds_scale_and_backend_commands():
    calls = []
    kc = KubectlCluster(context="shadowstack", namespace="default",
                        router="fleet-router", backend_index=1,
                        _run=lambda argv, **kw: calls.append(argv) or ("", 0))
    kc.serve_only("ornith-35b", isvc_all=["ornith-35b", "strix-coder"],
                  backend_for={"ornith-35b": "ornith-35b"})
    flat = " ".join(" ".join(c) for c in calls)
    assert "patch inferenceservice ornith-35b" in flat and '"replicas":1' in flat
    assert "patch inferenceservice strix-coder" in flat and '"replicas":0' in flat
    assert "patch modelrouter fleet-router" in flat
    assert "/spec/backends/1/inferenceServiceRef/name" in flat
