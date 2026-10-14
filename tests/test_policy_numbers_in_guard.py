"""Numbers the policy engine states (the computation and the §6 routing
reason) are numbers an agent may repeat.

Found in the 14 October Gemini evaluation: disp_ns_011's computation said
"the 5-min free wait", which the guard's minute pattern does not read as
5 min, so a Judge message saying "5 minutes" was rejected in one of two runs.
"""

from ridecourt.data import load_app_dataset
from ridecourt.guard import _extract_numbers
from ridecourt.policy import evaluate
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack


def test_every_threshold_in_computations_is_readable_by_the_guard():
    ds = load_app_dataset()
    for d in ds.disputes.values():
        trip = ds.trip(d.trip_id)
        facts = facts_from_evidence_pack(build_evidence_pack(d, ds))
        r = evaluate(d, trip, ds.profiles, facts, trip.fare)
        # a unit glued to a number with a hyphen is invisible to the guard
        assert "-min" not in r.computation, (d.dispute_id, r.computation)
        assert "-km" not in r.computation, (d.dispute_id, r.computation)


def test_routing_reason_numbers_are_extracted():
    nums = {(n.value, n.unit) for n in _extract_numbers("§6.2: route deviation 4.3 km > 2 km — human review", "")}
    assert (2.0, "km") in nums and (4.3, "km") in nums
