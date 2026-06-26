import json, subprocess, time
from typing import Callable, Dict, List, Optional, Protocol


class Cluster(Protocol):
    def serve_only(self, inference_service: str, isvc_all: List[str],
                   backend_for: Dict[str, str]) -> None: ...
    def wait_ready(self, inference_service: str, timeout_s: int = 600) -> None: ...
    def dispatch(self, agent: str, repo: str, issue: int, branch: str,
                 prompt: str, name: str) -> str: ...
    def wait_terminal(self, task: str, timeout_s: int = 3600) -> dict: ...
    def transcript_of(self, task: str) -> dict: ...
    def endpoint_port_forward(self, inference_service: str, local_port: int): ...


def _default_run(argv: List[str], stdin: Optional[str] = None):
    p = subprocess.run(argv, input=stdin, capture_output=True, text=True)
    return p.stdout, p.returncode


class KubectlCluster:
    def __init__(self, context, namespace, router, backend_index,
                 _run: Callable = _default_run):
        self.ctx, self.ns, self.router, self.bi = context, namespace, router, backend_index
        self._run = _run

    def _k(self, *args, stdin=None):
        return self._run(["kubectl", "--context", self.ctx, "-n", self.ns, *args], stdin=stdin)

    def serve_only(self, inference_service, isvc_all, backend_for):
        for isvc in isvc_all:
            replicas = 1 if isvc == inference_service else 0
            self._k("patch", "inferenceservice", isvc, "--type", "merge",
                    "-p", json.dumps({"spec": {"replicas": replicas}}, separators=(",", ":")))
        target = backend_for.get(inference_service, inference_service)
        self._run(["kubectl", "--context", self.ctx, "-n", self.ns, "patch", "modelrouter",
                   self.router, "--type", "json", "-p",
                   json.dumps([{"op": "replace",
                                "path": f"/spec/backends/{self.bi}/inferenceServiceRef/name",
                                "value": target}], separators=(",", ":"))])

    def wait_ready(self, inference_service, timeout_s=600):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            out, _ = self._k("get", "inferenceservice", inference_service,
                             "-o", "jsonpath={.status.phase}")
            if out.strip() == "Ready":
                return
            time.sleep(10)
        raise TimeoutError(f"{inference_service} not Ready in {timeout_s}s")

    def dispatch(self, agent, repo, issue, branch, prompt, name):
        manifest = json.dumps({
            "apiVersion": "foreman.llmkube.dev/v1alpha1", "kind": "AgenticTask",
            "metadata": {"name": name, "namespace": self.ns,
                         "labels": {"foreman.llmkube.dev/role": "coder", "bench": "agentic"}},
            "spec": {"kind": "issue-fix", "agentRef": {"name": agent}, "timeoutSeconds": 3300,
                     "payload": {"repo": repo, "issue": issue, "branch": branch, "prompt": prompt}},
        })
        self._run(["kubectl", "--context", self.ctx, "apply", "-f", "-"], stdin=manifest)
        return name

    def wait_terminal(self, task, timeout_s=3600):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            out, _ = self._k("get", "agentictask", task, "-o", "jsonpath={.status.phase}")
            if out.strip() in ("Succeeded", "Failed"):
                full, _ = self._k("get", "agentictask", task, "-o", "json")
                return json.loads(full).get("status", {})
            time.sleep(30)
        raise TimeoutError(f"task {task} did not terminate in {timeout_s}s")

    def transcript_of(self, task):
        out, rc = self._k("get", "cm", f"foreman-transcript-{task}", "-o", "json")
        if rc != 0:
            return {"turnCount": 0, "messages": []}
        return json.loads(json.loads(out)["data"]["transcript.json"])

    def endpoint_port_forward(self, inference_service, local_port):
        # returns a Popen the caller closes; not exercised in unit tests
        return subprocess.Popen(
            ["kubectl", "--context", self.ctx, "-n", self.ns, "port-forward",
             f"svc/{inference_service}", f"{local_port}:8080"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class FakeCluster:
    def __init__(self, terminal_status: dict, transcript: dict):
        self.terminal_status, self.transcript = terminal_status, transcript
        self.active: Optional[str] = None
        self.dispatched: List[str] = []

    def serve_only(self, inference_service, isvc_all, backend_for=None):
        self.active = inference_service

    def wait_ready(self, inference_service, timeout_s=600):
        pass

    def dispatch(self, agent, repo, issue, branch, prompt="", name=None):
        name = name or f"bench-{issue}"
        self.dispatched.append(name)
        return name

    def wait_terminal(self, task, timeout_s=3600):
        return self.terminal_status

    def transcript_of(self, task):
        return self.transcript

    def endpoint_port_forward(self, inference_service, local_port):
        class _NoopPF:
            def terminate(self_inner): pass
            def wait(self_inner, timeout=None): pass
        return _NoopPF()
