# W-Loop Batch 40: Methodology Paper Revision

Date: 2026-07-08
Role: W-Loop worker
Status: B47 repair pass completed in the workspace
Target: `research/methodology_paper.md`

Two invariants held:

1. Swing for the home run: the method must change how AI discovery claims are reviewed, not merely advise caution.
2. The loop only stops on a won-over adversary: B40 is written to give Q-Loop B48 a cleaner target, not to self-declare victory.

## Deliverables

| Artifact | Status | Purpose |
|---|---|---|
| `research/methodology_paper.md` | revised | shrunk B39 into a roster-relative, evidence-led methodology paper |
| `code/b40_positive_control.py` | added | small CPU-only positive-control runner |
| `experiments/b40_positive_control_measurement.json` | added | frozen output showing a narrow roster-relative signal |
| `research/work_loop_batch40.md` | added | 20-iteration work log |

## Positive-Control Result

The control used 12-bit inputs, 192 public training examples, 512 hidden cases, and a 1320-candidate three-feature interaction class.
The claimed interaction learner reached hidden HFA 1.0.
The best declared absorber reached 0.71875.
Component erasure dropped hidden HFA by 28.125 percentage points.
Randomized-label control reached 0.505859375.
The terminal token was:

```text
B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE
```

The control is deliberately bounded: the artifact records that full target-class PBE would absorb it if added to the roster.
That is the point of the revision: signal is real only relative to the declared roster and residual-risk statement.

## Twenty Iterations

| Iteration | Attack handled | Revision action | Result |
|---:|---|---|---|
| 1 | B47 I420 core claim overclaims | Replaced universal credibility language with roster-relative core claim in title block, abstract, Section 1, and conclusion. | Claim now says what the protocol can actually certify. |
| 2 | I409 no absorber-completeness stopping rule | Added Section 3 with stop conditions, status classes, and residual-risk bands. | Missing absorbers now lower claim ceiling instead of being hand-waved. |
| 3 | I408 no positive control | Built and ran `code/b40_positive_control.py`. | The ladder can be seen emitting a bounded `SIGNAL`. |
| 4 | I408 positive-control claim inflation risk | Recorded omitted full target-class PBE as high residual risk in the JSON and paper. | Positive control cannot be misused as universal discovery evidence. |
| 5 | I415 self-reference loop | Added Scope and Self-Audit Limits section. | Paper admits it is self-audited and lists independence work needed. |
| 6 | I407 negatives treated as validation | Split internal absorptions from methodology validation and added a separate positive-control role. | Case studies now support project honesty, not field validation. |
| 7 | I411 native absorber status implicit | Added filled case ledgers for C1-C4. | Each case now states rung status, evidence, cost note, and claim effect. |
| 8 | I417 appendices ask but do not answer | Converted checklist material into completed ledgers and compact rung/token cards. | The paper now answers its own hostile-review questions. |
| 9 | I416 appendix padding | Removed the generated B39 draft log and massive surface-by-rung matrix from the paper. | Paper shrank from 2,692 lines to about 460 rendered lines. |
| 10 | I410 cost fragility | Added Cost Robustness section with sensitivity bands. | B37 cheapness is robust; B38 cheapness is not overclaimed. |
| 11 | I413 B38 scale illusion | Added explicit measured-slice wording for 8 worlds, 256 cases, and 1536 predictions. | `2^64` is framed as search-space hardness, not evaluation-scale proof. |
| 12 | I412 scope exceeds evidence | Narrowed submission position to methodology proposal with internal evidence and a control. | No broad field-level claim without external validation. |
| 13 | I414 just good practice | Added Related-Work Delta table. | Contribution is terminal-token first refusal, not generic rigor. |
| 14 | I418 not actionable | Added eight-step domain onboarding workflow. | External evaluators get a concrete Monday-morning protocol. |
| 15 | I418 examples needed | Added mini-examples for game world models, theorem provers, and LLM hypotheses. | Workflow is less author-dependent. |
| 16 | I411 native absorber theater taxonomy only | Made status categories binding in the stopping rule and case ledgers. | Native/proxy/untested labels now affect tokens. |
| 17 | I410 homemade bit accounting | Required serialization, shared substrate split, side metrics, nominal ratios, and punitive bands. | Cost ledger is still imperfect but recomputable and claim-limited. |
| 18 | I419 no external-stakes case | Did not fake one; added external validation as missing work and narrowed pitch. | Paper obeys evidence instead of overreaching. |
| 19 | B47 final synthesis | Checked stale universal phrases and removed the old exact core-claim wording. | Revised paper no longer repeats the rejected claim except as a banned move. |
| 20 | Narrative gate | Ended the paper and this log with narrative sections. | Deliverables satisfy the mandatory narrative gate. |

## Validation

Commands run:

```text
python code/b40_positive_control.py --output experiments/b40_positive_control_measurement.json
(Get-Content research/methodology_paper.md | Measure-Object -Line -Word -Character) | Format-List
rg -n "not credible until|strongest boring explanations have failed|all ordinary explanations have failed|best non-discovery|adversarial measurement immune system|proved AI discovery" research/methodology_paper.md
ASCII scan over research/methodology_paper.md
```

Observed validation:

- Positive-control token: `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE`.
- Claimed hidden HFA: 1.0.
- Best declared absorber hidden HFA: 0.71875.
- Component-erasure drop: 28.125 percentage points.
- Revised paper rendered line count: about 460 by `rg` headings; PowerShell counted 365 nonblank-ish lines.
- Old paper line count before revision: 2,692 by PowerShell `Measure-Object`.
- ASCII scan: passed.
- Stale universal-claim search: only the banned-claim line remains, where the paper explicitly says it cannot certify all ordinary explanations failed.

## Commit Status

A logical first commit was attempted for the positive-control runner and JSON:

```text
git add code/b40_positive_control.py experiments/b40_positive_control_measurement.json
git commit -m "Add B40 positive-control measurement"
```

The sandbox blocked `.git` writes:

```text
fatal: Unable to create '.git/index.lock': Permission denied
```

Because `.git` is readable but not writable in this execution profile, commits could not be created from this session.
The workspace files are still written and ready for a local commit outside this sandbox.

## Residual Risks For B48

| Risk | Current handling |
|---|---|
| Positive control is synthetic and deliberately incomplete | marked as high residual risk, not field evidence |
| Internal case studies are self-authored | self-audit limits section narrows the claim |
| No external public claim reanalysis | listed as missing validation, not hidden |
| B38 cost ratio is marginal | described as comparable-cost absorption, not robust cheapness |
| Absorber rosters remain domain-judgment-heavy | onboarding recipe and stopping rule make judgment auditable |

## NARRATIVE SECTION

B40 turns the paper from an overconfident manifesto into a sharper protocol.
The ladder no longer claims to exhaust every ordinary explanation; it requires a declared roster, makes that roster executable, records what happened, and binds the public claim to residual risk.
That is the home-run version because it can change review behavior without cheating its own claim ceiling.