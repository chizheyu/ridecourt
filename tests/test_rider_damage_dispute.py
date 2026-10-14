"""§4.4: if the rider disputes a damage charge, the dispute goes to human
review. The first labels for these cases said "AI final", and the engine
agreed, so the label test could not catch it; the live pre-resolve showed two
rider damage disputes ruled by the AI."""

from ridecourt.data import load_app_dataset
from ridecourt.policy import evaluate
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack


def test_rider_damage_disputes_route_to_human():
    ds = load_app_dataset()
    cases = [d for d in ds.disputes.values()
             if d.category == "property_damage" and d.filed_by == "rider"]
    assert cases, "the synthetic set should contain rider damage disputes"
    for d in cases:
        trip = ds.trip(d.trip_id)
        facts = facts_from_evidence_pack(build_evidence_pack(d, ds))
        result = evaluate(d, trip, ds.profiles, facts, trip.fare)
        assert result.routing == "human", d.dispute_id
        assert "§4.4" in result.routing_reason, d.dispute_id
        assert ds.label(d.dispute_id).routing == "human", d.dispute_id
