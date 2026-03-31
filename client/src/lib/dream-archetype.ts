/**
 * dream-archetype.ts
 * Classifies dreams into universal archetypes based on symbols, themes,
 * and emotional arc using keyword scoring.
 *
 * No network calls. Works on data already stored in dreamAnalysis rows.
 */

export type DreamArchetype =
  | "chase_threat"
  | "flying"
  | "falling"
  | "search_quest"
  | "transformation"
  | "water_emotion"
  | "social_conflict"
  | "mechanical_work"
  | "nature_spiritual"
  | "death_rebirth";

export interface ArchetypeScore {
  archetype: DreamArchetype;
  /** Normalised 0–1 confidence */
  score: number;
  label: string;
  icon: string;
  description: string;
}

export interface DreamForArchetype {
  symbols: string[] | null;
  themes: string[] | null;
  emotionalArc: string | null;
}

// ── keyword maps ────────────────────────────────────────────────────────────

/** Each archetype has symbol keywords, theme keywords, and arc keywords. */
const ARCHETYPE_KEYWORDS: Record<
  DreamArchetype,
  { symbols: string[]; themes: string[]; arcs: string[] }
> = {
  chase_threat: {
    symbols: ["predator", "monster", "shadow", "weapon", "knife", "gun", "beast", "attacker", "danger", "threat", "pursuer", "enemy"],
    themes:  ["threat-simulation", "fear", "danger", "escape", "survival", "aggression"],
    arcs:    ["chased", "pursued", "fleeing", "escape", "terror", "fear", "panic", "threat", "danger"],
  },
  flying: {
    symbols: ["wings", "sky", "bird", "airplane", "cloud", "height", "air", "balloon", "flight", "soar"],
    themes:  ["freedom", "liberation", "elevation", "power", "transcendence"],
    arcs:    ["flying", "soaring", "floating", "freedom", "liberation", "elevation", "joy", "euphoria"],
  },
  falling: {
    symbols: ["cliff", "edge", "void", "abyss", "stairs", "bridge", "ladder", "ground", "floor"],
    themes:  ["loss-of-control", "anxiety", "failure", "collapse"],
    arcs:    ["falling", "plummeting", "slipping", "losing control", "collapse", "helpless", "sinking"],
  },
  search_quest: {
    symbols: ["map", "door", "key", "road", "path", "compass", "maze", "labyrinth", "mirror", "gate"],
    themes:  ["self-exploration", "journey", "discovery", "seeking", "identity"],
    arcs:    ["searching", "seeking", "lost", "exploring", "quest", "journey", "wandering", "looking"],
  },
  transformation: {
    symbols: ["caterpillar", "butterfly", "seed", "fire", "phoenix", "mirror", "metamorphosis", "mask", "shell"],
    themes:  ["transformation", "growth", "change", "rebirth", "integration", "transition"],
    arcs:    ["changing", "transforming", "growing", "evolving", "shifting", "becoming", "change"],
  },
  water_emotion: {
    symbols: ["ocean", "river", "lake", "rain", "flood", "wave", "water", "swimming", "drowning", "beach", "sea"],
    themes:  ["emotion", "unconscious", "depth", "flow", "cleansing", "overwhelm"],
    arcs:    ["drowning", "swimming", "floating", "submerged", "flood", "wave", "current", "swept"],
  },
  social_conflict: {
    symbols: ["crowd", "family", "stranger", "friend", "enemy", "argument", "table", "room", "school", "office"],
    themes:  ["relationships", "conflict", "social", "confrontation", "rejection", "belonging"],
    arcs:    ["arguing", "conflict", "confrontation", "rejected", "judged", "embarrassed", "social", "crowd"],
  },
  mechanical_work: {
    symbols: ["car", "train", "machine", "computer", "tool", "building", "phone", "clock", "vehicle", "engine"],
    themes:  ["work", "productivity", "control", "mechanism", "routine", "technology"],
    arcs:    ["driving", "working", "fixing", "breaking", "mechanical", "failing", "malfunction", "stuck"],
  },
  nature_spiritual: {
    symbols: ["forest", "tree", "mountain", "animal", "sun", "moon", "star", "garden", "stone", "light", "crystal"],
    themes:  ["nature", "spiritual", "connection", "peace", "transcendence", "lucid", "awe"],
    arcs:    ["peaceful", "serene", "awe", "wonder", "clarity", "spiritual", "transcendent", "calm"],
  },
  death_rebirth: {
    symbols: ["grave", "coffin", "skeleton", "ghost", "death", "funeral", "corpse", "spirit", "soul"],
    themes:  ["death", "endings", "loss", "grief", "renewal", "rebirth", "letting-go"],
    arcs:    ["dying", "death", "loss", "grief", "ending", "farewell", "rebirth", "renewal", "release"],
  },
};

export const ARCHETYPE_LABEL: Record<DreamArchetype, string> = {
  chase_threat:    "Chase / Threat",
  flying:          "Flying",
  falling:         "Falling",
  search_quest:    "Quest / Search",
  transformation:  "Transformation",
  water_emotion:   "Water / Emotion",
  social_conflict: "Social Conflict",
  mechanical_work: "Work / Machines",
  nature_spiritual:"Nature / Spirit",
  death_rebirth:   "Death / Rebirth",
};

export const ARCHETYPE_ICON: Record<DreamArchetype, string> = {
  chase_threat:    "🏃",
  flying:          "🦋",
  falling:         "⬇️",
  search_quest:    "🗺️",
  transformation:  "🌱",
  water_emotion:   "🌊",
  social_conflict: "👥",
  mechanical_work: "⚙️",
  nature_spiritual:"🌿",
  death_rebirth:   "🌙",
};

export const ARCHETYPE_DESCRIPTION: Record<DreamArchetype, string> = {
  chase_threat:    "Threat-simulation dreams; the mind rehearsing danger responses.",
  flying:          "Liberation and freedom; transcending limitations.",
  falling:         "Loss of control or anxiety; unsupported in waking life.",
  search_quest:    "Seeking identity, purpose, or missing pieces of the self.",
  transformation:  "Growth and change; the psyche processing transition.",
  water_emotion:   "The unconscious and emotional depths; being overwhelmed.",
  social_conflict: "Relationship dynamics, belonging, and social anxiety.",
  mechanical_work: "Work-related stress or problem-solving loops.",
  nature_spiritual:"Awe, peace, and connection to something larger.",
  death_rebirth:   "Endings and renewal; processing loss or major change.",
};

// ── scoring ──────────────────────────────────────────────────────────────────

const ALL_ARCHETYPES = Object.keys(ARCHETYPE_KEYWORDS) as DreamArchetype[];

function tokenise(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z\s-]/g, "").split(/\s+/).filter(Boolean);
}

function countMatches(tokens: string[], keywords: string[]): number {
  let hits = 0;
  for (const kw of keywords) {
    if (tokens.some((t) => t.includes(kw) || kw.includes(t))) hits++;
  }
  return hits;
}

/**
 * Score a single dream against every archetype.
 * Returns archetypes sorted by score descending, with score in [0,1].
 */
export function classifyDreamArchetypes(dream: DreamForArchetype): ArchetypeScore[] {
  const symbolTokens = (dream.symbols ?? []).flatMap((s) => tokenise(s));
  const themeTokens  = (dream.themes  ?? []).flatMap((t) => tokenise(t));
  const arcTokens    = dream.emotionalArc ? tokenise(dream.emotionalArc) : [];

  const scores = ALL_ARCHETYPES.map((archetype) => {
    const kw = ARCHETYPE_KEYWORDS[archetype];
    // Weights: symbols = 3, themes = 2, arc = 1
    const symScore  = countMatches(symbolTokens, kw.symbols) * 3;
    const thmScore  = countMatches(themeTokens,  kw.themes)  * 2;
    const arcScore  = countMatches(arcTokens,    kw.arcs)    * 1;
    const raw = symScore + thmScore + arcScore;
    return { archetype, raw };
  });

  const maxRaw = Math.max(...scores.map((s) => s.raw), 1);

  return scores
    .map(({ archetype, raw }) => ({
      archetype,
      score: raw / maxRaw,
      label: ARCHETYPE_LABEL[archetype],
      icon:  ARCHETYPE_ICON[archetype],
      description: ARCHETYPE_DESCRIPTION[archetype],
    }))
    .sort((a, b) => b.score - a.score);
}

// ── aggregation ──────────────────────────────────────────────────────────────

export interface AggregatedArchetype extends ArchetypeScore {
  /** Number of dreams where this archetype scored > 0 */
  dreamCount: number;
  /** Fraction of total dreams where this archetype was present */
  prevalence: number;
}

/**
 * Aggregate archetype scores across many dreams.
 * Returns archetypes sorted by summed score, limited to topN.
 */
export function aggregateArchetypes(
  dreams: DreamForArchetype[],
  topN = 5,
): AggregatedArchetype[] {
  if (dreams.length === 0) return [];

  const sumScore: Record<DreamArchetype, number>   = {} as Record<DreamArchetype, number>;
  const dreamCount: Record<DreamArchetype, number> = {} as Record<DreamArchetype, number>;

  for (const archetype of ALL_ARCHETYPES) {
    sumScore[archetype]   = 0;
    dreamCount[archetype] = 0;
  }

  for (const dream of dreams) {
    const scores = classifyDreamArchetypes(dream);
    for (const { archetype, score } of scores) {
      sumScore[archetype]   += score;
      if (score > 0) dreamCount[archetype]++;
    }
  }

  const totalDreams = dreams.length;

  return ALL_ARCHETYPES
    .map((archetype) => ({
      archetype,
      score:      totalDreams > 0 ? sumScore[archetype] / totalDreams : 0,
      dreamCount: dreamCount[archetype],
      prevalence: totalDreams > 0 ? dreamCount[archetype] / totalDreams : 0,
      label:      ARCHETYPE_LABEL[archetype],
      icon:       ARCHETYPE_ICON[archetype],
      description: ARCHETYPE_DESCRIPTION[archetype],
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

/**
 * Dominant archetype for a single dream (highest-scoring, or null if all zero).
 */
export function dominantArchetype(dream: DreamForArchetype): ArchetypeScore | null {
  const scores = classifyDreamArchetypes(dream);
  const top = scores[0];
  return top && top.score > 0 ? top : null;
}
