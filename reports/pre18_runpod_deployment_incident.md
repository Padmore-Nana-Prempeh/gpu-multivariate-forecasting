# Pre-18 RunPod A40 Deployment Incident Report

**Project:** GPU-Accelerated Multivariate Forecasting

**Incident date:** November 6, 2026

**Status:** Pod stopped; root cause undetermined; human support investigation pending

**RunPod support ticket:** #55188

## 1. Executive Summary

A custom Linux AMD64 Docker image was built and validated locally, published successfully to GitHub Container Registry (GHCR), and configured for deployment on an NVIDIA A40 GPU through RunPod.

The RunPod deployment request was accepted, but the Pod remained in initialization and never became accessible for GPU preflight testing.

System logs repeatedly showed `create container` events. Container logs displayed `No logs found`.

The Pod was stopped to limit further compute charges. The 50 GB network volume was retained.

No official Phase 18 training experiments were launched.

The underlying cause remains unknown. Registry authentication, image downloading, container creation, and host infrastructure have not yet been ruled out.

## 2. Source and Docker Provenance

| Property | Value |
|---|---|
| Repository | Padmore-Nana-Prempeh/gpu-multivariate-forecasting |
| Docker environment commit | `ea2c8f9` |
| GHCR publication evidence commit | `36a9667` |
| Operating system | Linux AMD64, Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA toolkit | 12.8 |
| CUDA compiler | nvcc 12.8.93 |
| Python dependency lock | 40 installed packages |
| CPU regression tests | 26 passed |
| Official experiment configurations | 60 verified |

**Published registry reference:**

`ghcr.io/padmore-nana-prempeh/gpu-multivariate-forecasting:pre18-ea2c8f9`

**OCI image index digest:**

`sha256:9b5b9d135c4ddd16c4df4b045a87836428b7ab3fe58b5992cc4ab419483d6d1d`

**Linux AMD64 manifest digest:**

`sha256:589cb1cd2bacbee1e539cc091ef0de08faba6d635d2feedd5044adefa3766db9`

The registry publication and digest were independently verified. The local environment lock, regression tests, and safe default runner had passed before cloud deployment.

Local validation does not establish successful execution on RunPod hardware.

## 3. RunPod Deployment Configuration

| Property | Configuration |
|---|---|
| Pod name | `overwhelming_red_roadrunner` |
| Pod ID | `hwusdlb5p9wqz7` |
| Template | `gpuforecast-a40-pre18` |
| Data center | CA-MTL-1, Canada |
| GPU | 1 NVIDIA A40, 48 GB VRAM |
| CPU/RAM | 9 vCPU / 50 GB RAM |
| Container disk | 60 GB |
| Network volume | `gpuforecast-a40-results` |
| Network volume capacity | 50 GB |
| Network volume mount | `/runpod-volume` |
| Storage tier | High-performance network storage |
| Registry credential | `ghcr-gpuforecast-readonly` |
| Startup command | `sleep infinity` |

The registry credential was registered in RunPod and selected in the template. This confirms the configuration, not whether backend authentication succeeded.

The image was referenced by its published OCI index digest.

The separate network volume mount was chosen to avoid hiding project files already present under `/workspace`.

## 4. Incident Timeline

All times are approximate and refer to November 6, 2026.

- **Before deployment:** Local Docker validation, GHCR publication, and template creation completed.
- **Approximately 12:00 PM Eastern:** RunPod began creating the container.
- **12:00:46–12:04:55 PM Eastern:** Captured system logs repeatedly displayed `create container` events, approximately every 15 seconds.
- **During initialization:** Container logs displayed `No logs found`; no usable terminal was available.
- **Approximately 12:16 PM Eastern:** Pod confirmed stopped, with compute pricing displayed as $0.00/hour.
- **Later that day:** RunPod support ticket #55188 was created and acknowledged by email.

The approximate investigation window provided to support was 17:00–17:16 UTC.

## 5. Observed Failure

The following observations were supported by RunPod screenshots:

1. The deployment request was accepted and a Pod was created.
2. The Pod remained initializing.
3. System logs repeated container-creation events.
4. Container logs contained no application output.
5. No successful container start or accessible terminal was observed.
6. The Pod was subsequently stopped.

The available evidence does not identify the exact failed lifecycle operation.

Potential causes requiring investigation include:

- Private registry authentication or credential binding.
- GHCR image download or extraction.
- Container runtime initialization.
- Host-side infrastructure or scheduling problems.

These are diagnostic possibilities, not confirmed findings.

## 6. Compute Cost and Storage

At deployment, RunPod displayed approximately $0.60/hour for the selected A40 Pod and container disk.

After stopping, the Pod details displayed $0.00/hour for compute and container storage.

The network volume remained allocated separately at a displayed rate of approximately $7/month.

The exact amount charged during unsuccessful initialization has not been independently reconciled. A review of those charges was requested from RunPod support.

## 7. Impact on the Research Project

The physical NVIDIA A40 preflight could not be completed.

Therefore, the following remain unverified in the deployed container:

- NVIDIA driver and CUDA runtime interoperability.
- Actual A40 device discovery.
- GPU tensor correctness.
- Custom CUDA extension compilation and execution.
- Container access to the mounted dataset.
- GPU model smoke tests.
- Persistent checkpoint and results writes.

**Official Phase 18 campaign status:** 0 of 60 runs executed.

The frozen experiment configurations, failure-safe runner, Docker environment, and published registry image remain preserved.

## 8. Support Escalation

**Ticket:** #55188

**Subject:** NVIDIA A40 Pod stuck initializing - GHCR image

RunPod acknowledged receipt and indicated that the request had been sent to its human support team.

The support request asks RunPod to:

1. Examine backend registry-pull and container-creation events.
2. Determine the underlying failure.
3. Provide a verified corrective action before redeployment.
4. Review compute charges incurred while the Pod was unusable.

The automated support response discussed possible causes but did not supply a confirmed root cause.

## 9. Recovery Plan

Do not modify the frozen official experiment matrix or start the campaign while the deployment problem remains unresolved.

Before another paid deployment:

1. Review the human support response and identify the confirmed issue.
2. Apply only necessary corrections, documenting any configuration or image changes.
3. Verify that the private GHCR image is accessible to the selected RunPod configuration.
4. Confirm storage attachment, startup command, and an available terminal-access method.
5. Start a controlled A40 preflight and verify physical CUDA execution.
6. Test persistent dataset and results paths.
7. Execute short GPU correctness and smoke tests.
8. Record evidence and commit the preflight results.
9. Authorize the official 60-run campaign only after all relevant checks pass.

## 10. Current Decision

The RunPod Pod remains stopped.

The previously created network volume is retained, subject to ongoing storage charges.

No further paid deployment is planned until a specific diagnostic finding or well-defined corrective test is available.

**Incident status:** Open, awaiting RunPod support findings.

---

## Evidence References

- `reports/evidence/pre18/docker_validation.log`
- `reports/evidence/pre18/image_provenance.txt`
- `reports/evidence/pre18/ghcr_publication.txt`
- RunPod system-log screenshots captured November 6, 2026.
- RunPod container-log screenshot showing `No logs found`.
- RunPod stopped-Pod details screenshots.
- RunPod support acknowledgment, ticket #55188.

**Evidence limitation:** The RunPod screenshots and support email are conversation evidence. They are not represented as repository files unless separately added and verified.
