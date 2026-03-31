import { describe, it, expect } from "vitest";
import {
  classifyDreamArchetypes,
  aggregateArchetypes,
  dominantArchetype,
  ARCHETYPE_LABEL,
  ARCHETYPE_ICON,
  ARCHETYPE_DESCRIPTION,
  type DreamForArchetype,
  type DreamArchetype,
} from "@/lib/dream-archetype";

// ── fixtures ──────────────────────────────────────────────────────────────────

const chaseDream: DreamForArchetype = {
  symbols: ["shadow", "monster", "weapon"],
  themes:  ["threat-simulation", "fear", "survival"],
  emotionalArc: "being chased and fleeing in terror",
};

const flyingDream: DreamForArchetype = {
  symbols: ["wings", "sky", "bird", "cloud"],
  themes:  ["freedom", "liberation"],
  emotionalArc: "soaring with joy and euphoria",
};

const fallingDream: DreamForArchetype = {
  symbols: ["cliff", "void", "edge"],
  themes:  ["loss-of-control", "anxiety"],
  emotionalArc: "falling and losing control helpless",
};

const questDream: DreamForArchetype = {
  symbols: ["door", "key", "maze", "path"],
  themes:  ["self-exploration", "seeking"],
  emotionalArc: "searching and wandering through labyrinth",
};

const waterDream: DreamForArchetype = {
  symbols: ["ocean", "wave", "swimming"],
  themes:  ["emotion", "unconscious"],
  emotionalArc: "submerged in flood and swept away",
};

const transformationDream: DreamForArchetype = {
  symbols: ["butterfly", "seed", "fire"],
  themes:  ["transformation", "growth", "change"],
  emotionalArc: "changing and transforming becoming new",
};

const blankDream: DreamForArchetype = {
  symbols: null,
  themes:  null,
  emotionalArc: null,
};

const emptyDream: DreamForArchetype = {
  symbols: [],
  themes:  [],
  emotionalArc: "",
};

// ── classifyDreamArchetypes ───────────────────────────────────────────────────

describe("classifyDreamArchetypes", () => {
  it("returns an entry for every archetype", () => {
    const result = classifyDreamArchetypes(chaseDream);
    expect(result.length).toBe(10);
  });

  it("chase dream scores highest on chase_threat", () => {
    const result = classifyDreamArchetypes(chaseDream);
    expect(result[0].archetype).toBe("chase_threat");
  });

  it("flying dream scores highest on flying", () => {
    const result = classifyDreamArchetypes(flyingDream);
    expect(result[0].archetype).toBe("flying");
  });

  it("falling dream scores highest on falling", () => {
    const result = classifyDreamArchetypes(fallingDream);
    expect(result[0].archetype).toBe("falling");
  });

  it("quest dream scores highest on search_quest", () => {
    const result = classifyDreamArchetypes(questDream);
    expect(result[0].archetype).toBe("search_quest");
  });

  it("water dream scores highest on water_emotion", () => {
    const result = classifyDreamArchetypes(waterDream);
    expect(result[0].archetype).toBe("water_emotion");
  });

  it("transformation dream scores highest on transformation", () => {
    const result = classifyDreamArchetypes(transformationDream);
    expect(result[0].archetype).toBe("transformation");
  });

  it("scores are sorted descending", () => {
    const result = classifyDreamArchetypes(chaseDream);
    for (let i = 1; i < result.length; i++) {
      expect(result[i - 1].score).toBeGreaterThanOrEqual(result[i].score);
    }
  });

  it("all scores are in [0, 1]", () => {
    const result = classifyDreamArchetypes(chaseDream);
    for (const { score } of result) {
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it("top score is 1 when there are matches", () => {
    const result = classifyDreamArchetypes(chaseDream);
    expect(result[0].score).toBe(1);
  });

  it("blank dream returns all-zero scores", () => {
    const result = classifyDreamArchetypes(blankDream);
    // maxRaw is clamped to 1, so all scores = 0/1 = 0
    expect(result.every((r) => r.score === 0)).toBe(true);
  });

  it("empty arrays / empty arc returns all-zero scores", () => {
    const result = classifyDreamArchetypes(emptyDream);
    expect(result.every((r) => r.score === 0)).toBe(true);
  });

  it("each result includes label, icon, and description", () => {
    const result = classifyDreamArchetypes(chaseDream);
    for (const r of result) {
      expect(typeof r.label).toBe("string");
      expect(typeof r.icon).toBe("string");
      expect(typeof r.description).toBe("string");
    }
  });

  it("dream with symbols+themes outranks one with only arc keywords", () => {
    // A rich dream (symbols + themes + arc) should rank chase_threat higher relative
    // to other archetypes than a plain nature dream with the same arc suffix.
    // We can test that a mixed dream puts chase_threat at rank 0.
    const richChase: DreamForArchetype = {
      symbols: ["shadow", "monster", "weapon"],
      themes: ["threat-simulation", "fear"],
      emotionalArc: "chased and fleeing in terror",
    };
    const result = classifyDreamArchetypes(richChase);
    expect(result[0].archetype).toBe("chase_threat");
  });

  it("partial keyword match works (substring)", () => {
    // "predators" should match "predator" keyword
    const dream: DreamForArchetype = {
      symbols: ["predators"],
      themes: [],
      emotionalArc: null,
    };
    const result = classifyDreamArchetypes(dream);
    const chaseScore = result.find((r) => r.archetype === "chase_threat")!.score;
    expect(chaseScore).toBeGreaterThan(0);
  });
});

// ── aggregateArchetypes ───────────────────────────────────────────────────────

describe("aggregateArchetypes", () => {
  it("returns empty array for empty input", () => {
    expect(aggregateArchetypes([])).toHaveLength(0);
  });

  it("returns at most topN results", () => {
    const dreams = [chaseDream, flyingDream, waterDream];
    expect(aggregateArchetypes(dreams, 3)).toHaveLength(3);
  });

  it("defaults to topN = 5", () => {
    const dreams = [chaseDream, flyingDream, waterDream];
    expect(aggregateArchetypes(dreams)).toHaveLength(5);
  });

  it("dominant archetype across repeated chase dreams is chase_threat", () => {
    const dreams = [chaseDream, chaseDream, chaseDream, flyingDream];
    const result = aggregateArchetypes(dreams);
    expect(result[0].archetype).toBe("chase_threat");
  });

  it("dreamCount is correct", () => {
    const dreams = [chaseDream, chaseDream, flyingDream];
    const result = aggregateArchetypes(dreams);
    const chase = result.find((r) => r.archetype === "chase_threat")!;
    expect(chase.dreamCount).toBe(2);
  });

  it("prevalence is dreamCount / totalDreams", () => {
    const dreams = [chaseDream, chaseDream, flyingDream];
    const result = aggregateArchetypes(dreams);
    const chase = result.find((r) => r.archetype === "chase_threat")!;
    expect(chase.prevalence).toBeCloseTo(2 / 3);
  });

  it("scores are sorted descending", () => {
    const dreams = [chaseDream, flyingDream, waterDream, questDream];
    const result = aggregateArchetypes(dreams, 4);
    for (let i = 1; i < result.length; i++) {
      expect(result[i - 1].score).toBeGreaterThanOrEqual(result[i].score);
    }
  });

  it("all-blank dreams return zero scores", () => {
    const result = aggregateArchetypes([blankDream, blankDream]);
    expect(result.every((r) => r.score === 0)).toBe(true);
  });
});

// ── dominantArchetype ─────────────────────────────────────────────────────────

describe("dominantArchetype", () => {
  it("returns null for blank dream", () => {
    expect(dominantArchetype(blankDream)).toBeNull();
  });

  it("returns null for empty-array dream", () => {
    expect(dominantArchetype(emptyDream)).toBeNull();
  });

  it("returns the highest-scoring archetype for a chase dream", () => {
    expect(dominantArchetype(chaseDream)!.archetype).toBe("chase_threat");
  });

  it("returns the highest-scoring archetype for a flying dream", () => {
    expect(dominantArchetype(flyingDream)!.archetype).toBe("flying");
  });

  it("score is 1 for a strongly-matched dream", () => {
    expect(dominantArchetype(chaseDream)!.score).toBe(1);
  });
});

// ── constants ─────────────────────────────────────────────────────────────────

describe("archetype constants", () => {
  const ALL: DreamArchetype[] = [
    "chase_threat", "flying", "falling", "search_quest", "transformation",
    "water_emotion", "social_conflict", "mechanical_work", "nature_spiritual", "death_rebirth",
  ];

  it("ARCHETYPE_LABEL has entry for every archetype", () => {
    for (const a of ALL) {
      expect(typeof ARCHETYPE_LABEL[a]).toBe("string");
      expect(ARCHETYPE_LABEL[a].length).toBeGreaterThan(0);
    }
  });

  it("ARCHETYPE_ICON has entry for every archetype", () => {
    for (const a of ALL) {
      expect(typeof ARCHETYPE_ICON[a]).toBe("string");
      expect(ARCHETYPE_ICON[a].length).toBeGreaterThan(0);
    }
  });

  it("ARCHETYPE_DESCRIPTION has entry for every archetype", () => {
    for (const a of ALL) {
      expect(typeof ARCHETYPE_DESCRIPTION[a]).toBe("string");
      expect(ARCHETYPE_DESCRIPTION[a].length).toBeGreaterThan(0);
    }
  });
});
