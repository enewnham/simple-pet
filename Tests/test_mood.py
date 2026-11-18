from pet_daemon.behavior.mood import (
    EVENT_HISTORY_LIMIT,
    IdlePhase,
    Mood,
    MoodCue,
    MoodState,
)
from Tests.test_behavior import Clock


def test_recovery_cue_settles_distress_and_replay_has_no_effect_after_restart() -> None:
    clock = Clock()
    mood = Mood(clock)
    assert mood.observe(MoodCue.STRUGGLE, "struggle:testing")
    assert mood.snapshot().agitation > 0.4
    assert mood.observe(MoodCue.RECOVERY, "recovery:testing")
    recovered = mood.checkpoint()
    assert recovered.agitation < 0.1

    restored = Mood(
        clock, state=MoodState.model_validate_json(recovered.model_dump_json())
    )
    assert not restored.observe(MoodCue.STRUGGLE, "struggle:testing")
    assert restored.snapshot().agitation == recovered.agitation


def test_idle_presence_degrades_despite_work_and_resets_on_engagement() -> None:
    clock = Clock()
    mood = Mood(clock)
    mood.observe_work(True)
    clock.advance(100)
    assert mood.work_quiet_age == 0
    assert mood.engagement_age == 100
    assert mood.claim_presence()
    assert not mood.claim_presence()
    mood.observe_work(True, changed=True)
    assert not mood.claim_presence()

    clock.advance(500)
    assert mood.idle_phase == IdlePhase.SETTLED
    assert not mood.claim_presence()
    assert mood.interact("click")
    assert mood.idle_phase == IdlePhase.FRESH
    assert mood.engagement_age == 0
    assert mood.novelty_age == 600


def test_restart_decays_mood_without_replaying_idle_presence() -> None:
    clock = Clock()
    mood = Mood(clock)
    mood.observe(MoodCue.STRUGGLE, "struggle")
    clock.advance(100)
    state = mood.checkpoint()
    clock.advance(3600)
    resumed = Mood(clock, state=state)

    assert resumed.idle_phase == IdlePhase.SETTLED
    assert not resumed.claim_presence()
    assert resumed.engagement_age == 3700
    assert resumed.snapshot().agitation < state.agitation


def test_an_unanswered_bid_affects_mood_once() -> None:
    clock = Clock()
    mood = Mood(clock)
    clock.advance(100)
    assert mood.claim_presence()
    clock.advance(100)
    assert mood.observe(MoodCue.UNANSWERED, "unanswered")
    assert mood.emotion == "a little vexed"
    assert not mood.observe(MoodCue.UNANSWERED, "unanswered")
    assert not mood.claim_presence()
    assert mood.interact("click")
    assert mood.emotion == "at ease"


def test_wake_reconciles_wall_time_when_monotonic_clock_stops() -> None:
    wall = Clock()
    mood = Mood(wall)
    assert mood.observe(MoodCue.STRUGGLE, "failure")
    mood.suspend()
    wall.advance(3600)
    mood.resume()

    assert mood.engagement_age == 3600
    assert mood.idle_phase == IdlePhase.SETTLED
    assert mood.checkpoint().updated_at == wall()
    assert not mood.claim_presence()
    assert mood.snapshot().agitation < 0.1
    assert mood.interact("return", observed_at=wall())
    assert mood.engagement_age == 0


def test_retracting_declared_engagement_restores_local_state_and_budget() -> None:
    clock = Clock()
    mood = Mood(clock)
    clock.advance(500)
    assert mood.claim_presence()
    local_appetite = mood.snapshot().social_appetite
    local_engagement = mood.last_local_engagement_at

    assert mood.set_external_engagement(clock())
    assert mood.idle_phase == IdlePhase.FRESH
    assert mood.snapshot().social_appetite < local_appetite
    assert mood.checkpoint().social_appetite == local_appetite
    assert mood.last_local_engagement_at == local_engagement

    assert mood.set_external_engagement(None)
    assert mood.engagement_age == 500
    assert mood.snapshot().social_appetite == local_appetite
    assert not mood.claim_presence()


def test_checkpoint_keeps_bounded_event_ids_and_deduplicates_on_restart() -> None:
    clock = Clock()
    mood = Mood(clock)
    for index in range(EVENT_HISTORY_LIMIT + 1):
        assert mood.observe(MoodCue.NOVELTY, str(index))

    stored = mood.checkpoint()
    assert len(stored.applied_events) == EVENT_HISTORY_LIMIT
    restored = Mood(clock, state=stored)
    assert not restored.observe(MoodCue.NOVELTY, str(EVENT_HISTORY_LIMIT))
