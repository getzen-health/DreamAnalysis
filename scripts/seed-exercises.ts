/**
 * Seed script: inserts 200+ exercises into the exercises table.
 *
 * Usage:
 *   node --import tsx/esm scripts/seed-exercises.ts
 *
 * Safe to run multiple times — uses ON CONFLICT (name) DO NOTHING.
 */

import postgres from "postgres";

const DATABASE_URL =
  "postgresql://postgres:neuraldream2026@db.tpiyavugafhplsmwvrel.supabase.co:5432/postgres";

interface Exercise {
  name: string;
  category: "strength" | "cardio" | "flexibility" | "hiit";
  muscle_groups: string[];
  equipment: string;
  instructions: string;
}

const exercises: Exercise[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — CHEST (18)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Barbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "barbell", instructions: "Lie on a flat bench, grip the barbell slightly wider than shoulder-width, lower to chest and press up." },
  { name: "Incline Barbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "barbell", instructions: "Set bench to 30-45 degrees, grip barbell and lower to upper chest, then press up." },
  { name: "Decline Barbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "barbell", instructions: "Lie on a decline bench, grip barbell and lower to lower chest, then press up." },
  { name: "Dumbbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "dumbbell", instructions: "Lie on a flat bench holding dumbbells at chest level, press up until arms are extended." },
  { name: "Incline Dumbbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "dumbbell", instructions: "Set bench to 30-45 degrees, press dumbbells from upper chest to full extension." },
  { name: "Decline Dumbbell Bench Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "dumbbell", instructions: "Lie on a decline bench, press dumbbells from lower chest to full extension." },
  { name: "Dumbbell Chest Fly", category: "strength", muscle_groups: ["chest"], equipment: "dumbbell", instructions: "Lie on a flat bench with arms extended, lower dumbbells in a wide arc until you feel a stretch, then bring back together." },
  { name: "Incline Dumbbell Fly", category: "strength", muscle_groups: ["chest", "shoulders"], equipment: "dumbbell", instructions: "Set bench to 30-45 degrees, perform a fly motion with slight elbow bend, squeezing at the top." },
  { name: "Cable Crossover", category: "strength", muscle_groups: ["chest"], equipment: "cable", instructions: "Stand between cable towers set high, pull handles down and together in front of your chest in an arc motion." },
  { name: "Low Cable Crossover", category: "strength", muscle_groups: ["chest", "shoulders"], equipment: "cable", instructions: "Set cables at the lowest position, pull handles up and together in front of your chest." },
  { name: "Push-Up", category: "strength", muscle_groups: ["chest", "triceps", "shoulders", "core"], equipment: "bodyweight", instructions: "Start in plank position, lower your body until chest nearly touches the floor, then push back up." },
  { name: "Wide Push-Up", category: "strength", muscle_groups: ["chest", "shoulders"], equipment: "bodyweight", instructions: "Perform a push-up with hands placed wider than shoulder-width to emphasize chest engagement." },
  { name: "Diamond Push-Up", category: "strength", muscle_groups: ["chest", "triceps"], equipment: "bodyweight", instructions: "Place hands together forming a diamond shape under your chest, lower and press back up." },
  { name: "Decline Push-Up", category: "strength", muscle_groups: ["chest", "shoulders", "triceps"], equipment: "bodyweight", instructions: "Place feet on an elevated surface, perform push-ups to target the upper chest." },
  { name: "Chest Dip", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "bodyweight", instructions: "On parallel bars, lean forward slightly and lower your body until upper arms are parallel to the floor, then press up." },
  { name: "Machine Chest Press", category: "strength", muscle_groups: ["chest", "triceps", "shoulders"], equipment: "machine", instructions: "Sit in the machine, grip handles at chest height, and press forward until arms are extended." },
  { name: "Pec Deck Machine", category: "strength", muscle_groups: ["chest"], equipment: "machine", instructions: "Sit with arms on pads at chest height, squeeze pads together in front of your chest." },
  { name: "Svend Press", category: "strength", muscle_groups: ["chest"], equipment: "dumbbell", instructions: "Hold a weight plate between your palms at chest height and press forward, squeezing your chest." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — BACK (16)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Conventional Deadlift", category: "strength", muscle_groups: ["back", "glutes", "hamstrings", "core"], equipment: "barbell", instructions: "Stand with feet hip-width, grip barbell, drive through heels and lift by extending hips and knees simultaneously." },
  { name: "Sumo Deadlift", category: "strength", muscle_groups: ["back", "glutes", "hamstrings", "quads"], equipment: "barbell", instructions: "Take a wide stance with toes pointed out, grip barbell between legs, and lift by extending hips and knees." },
  { name: "Barbell Bent-Over Row", category: "strength", muscle_groups: ["back", "biceps", "rear delts"], equipment: "barbell", instructions: "Hinge at hips with slight knee bend, pull barbell to lower chest while keeping back flat." },
  { name: "Dumbbell Bent-Over Row", category: "strength", muscle_groups: ["back", "biceps"], equipment: "dumbbell", instructions: "Hinge at hips holding dumbbells, pull to hip level squeezing shoulder blades together." },
  { name: "Single-Arm Dumbbell Row", category: "strength", muscle_groups: ["back", "biceps"], equipment: "dumbbell", instructions: "Place one knee and hand on a bench, row a dumbbell to your hip with the free arm." },
  { name: "Pull-Up", category: "strength", muscle_groups: ["back", "biceps", "core"], equipment: "bodyweight", instructions: "Hang from a bar with overhand grip, pull yourself up until chin clears the bar." },
  { name: "Chin-Up", category: "strength", muscle_groups: ["back", "biceps"], equipment: "bodyweight", instructions: "Hang from a bar with underhand grip, pull yourself up until chin clears the bar." },
  { name: "Lat Pulldown", category: "strength", muscle_groups: ["back", "biceps"], equipment: "cable", instructions: "Sit at the lat pulldown machine, grip the bar wide, and pull down to upper chest." },
  { name: "Close-Grip Lat Pulldown", category: "strength", muscle_groups: ["back", "biceps"], equipment: "cable", instructions: "Use a V-bar or close-grip handle, pull down to upper chest focusing on lower lats." },
  { name: "Seated Cable Row", category: "strength", muscle_groups: ["back", "biceps", "rear delts"], equipment: "cable", instructions: "Sit upright at a cable row station, pull the handle to your midsection, squeezing shoulder blades." },
  { name: "T-Bar Row", category: "strength", muscle_groups: ["back", "biceps", "rear delts"], equipment: "barbell", instructions: "Straddle the barbell with a V-grip handle, hinge at hips and row the bar to your chest." },
  { name: "Face Pull", category: "strength", muscle_groups: ["rear delts", "traps", "rotator cuff"], equipment: "cable", instructions: "Set cable at face height with rope attachment, pull toward your face with elbows high and externally rotate." },
  { name: "Pendlay Row", category: "strength", muscle_groups: ["back", "biceps", "core"], equipment: "barbell", instructions: "Start each rep with the barbell on the floor, explosively row to lower chest, then lower back to the floor." },
  { name: "Inverted Row", category: "strength", muscle_groups: ["back", "biceps", "core"], equipment: "bodyweight", instructions: "Lie under a bar set at waist height, grip with overhand grip, pull chest to the bar keeping body straight." },
  { name: "Straight-Arm Pulldown", category: "strength", muscle_groups: ["back"], equipment: "cable", instructions: "Stand facing a high cable with straight arms, pull the bar down to your thighs while keeping arms extended." },
  { name: "Rack Pull", category: "strength", muscle_groups: ["back", "traps", "glutes"], equipment: "barbell", instructions: "Set barbell at knee height in a rack, grip and stand up by extending hips, focusing on upper back squeeze." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — SHOULDERS (14)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Barbell Overhead Press", category: "strength", muscle_groups: ["shoulders", "triceps", "core"], equipment: "barbell", instructions: "Stand with barbell at shoulder height, press overhead until arms are fully extended." },
  { name: "Dumbbell Overhead Press", category: "strength", muscle_groups: ["shoulders", "triceps"], equipment: "dumbbell", instructions: "Sit or stand with dumbbells at shoulder height, press up until arms are extended overhead." },
  { name: "Seated Dumbbell Shoulder Press", category: "strength", muscle_groups: ["shoulders", "triceps"], equipment: "dumbbell", instructions: "Sit on a bench with back support, press dumbbells from shoulder height to full extension overhead." },
  { name: "Arnold Press", category: "strength", muscle_groups: ["shoulders", "triceps"], equipment: "dumbbell", instructions: "Start with dumbbells at chest with palms facing you, rotate palms outward as you press overhead." },
  { name: "Dumbbell Lateral Raise", category: "strength", muscle_groups: ["shoulders"], equipment: "dumbbell", instructions: "Stand with dumbbells at sides, raise arms out to the sides until parallel with the floor." },
  { name: "Cable Lateral Raise", category: "strength", muscle_groups: ["shoulders"], equipment: "cable", instructions: "Stand beside a low cable, grip the handle with the far hand and raise arm out to shoulder height." },
  { name: "Dumbbell Front Raise", category: "strength", muscle_groups: ["shoulders"], equipment: "dumbbell", instructions: "Stand with dumbbells in front of thighs, raise one or both arms forward to shoulder height." },
  { name: "Rear Delt Fly", category: "strength", muscle_groups: ["rear delts", "traps"], equipment: "dumbbell", instructions: "Bend forward at the hips, raise dumbbells out to the sides squeezing your shoulder blades." },
  { name: "Cable Rear Delt Fly", category: "strength", muscle_groups: ["rear delts", "traps"], equipment: "cable", instructions: "Set cables at shoulder height, cross the cables and pull outward to each side with straight arms." },
  { name: "Upright Row", category: "strength", muscle_groups: ["shoulders", "traps"], equipment: "barbell", instructions: "Hold barbell with narrow grip in front of thighs, pull up along your body to chin height, elbows flaring out." },
  { name: "Dumbbell Shrug", category: "strength", muscle_groups: ["traps"], equipment: "dumbbell", instructions: "Hold dumbbells at your sides, shrug shoulders straight up toward ears and hold briefly." },
  { name: "Barbell Shrug", category: "strength", muscle_groups: ["traps"], equipment: "barbell", instructions: "Hold barbell in front of thighs, shrug shoulders straight up toward ears." },
  { name: "Machine Shoulder Press", category: "strength", muscle_groups: ["shoulders", "triceps"], equipment: "machine", instructions: "Sit in the machine, grip handles at shoulder height, and press overhead." },
  { name: "Landmine Press", category: "strength", muscle_groups: ["shoulders", "chest", "core"], equipment: "barbell", instructions: "Hold the end of a barbell anchored on the floor at shoulder height, press upward and forward." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — BICEPS (10)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Barbell Bicep Curl", category: "strength", muscle_groups: ["biceps"], equipment: "barbell", instructions: "Stand with barbell at arm's length, curl the bar up to shoulder height without swinging." },
  { name: "Dumbbell Bicep Curl", category: "strength", muscle_groups: ["biceps"], equipment: "dumbbell", instructions: "Stand with dumbbells at sides, curl both or alternating up to shoulder height." },
  { name: "Hammer Curl", category: "strength", muscle_groups: ["biceps", "forearms"], equipment: "dumbbell", instructions: "Hold dumbbells with neutral grip (palms facing each other), curl up keeping wrists neutral." },
  { name: "Preacher Curl", category: "strength", muscle_groups: ["biceps"], equipment: "dumbbell", instructions: "Rest upper arms on a preacher bench pad, curl the weight up from full extension." },
  { name: "Concentration Curl", category: "strength", muscle_groups: ["biceps"], equipment: "dumbbell", instructions: "Sit on a bench, brace elbow against inner thigh, curl dumbbell up with strict form." },
  { name: "EZ-Bar Curl", category: "strength", muscle_groups: ["biceps"], equipment: "barbell", instructions: "Use an EZ-curl bar for a more natural wrist angle, curl from full extension to shoulder height." },
  { name: "Cable Bicep Curl", category: "strength", muscle_groups: ["biceps"], equipment: "cable", instructions: "Stand facing a low cable with straight bar attachment, curl up to shoulder height." },
  { name: "Incline Dumbbell Curl", category: "strength", muscle_groups: ["biceps"], equipment: "dumbbell", instructions: "Sit on an incline bench (45 degrees), let arms hang and curl dumbbells up for a greater stretch." },
  { name: "Spider Curl", category: "strength", muscle_groups: ["biceps"], equipment: "dumbbell", instructions: "Lie chest-down on an incline bench, let arms hang straight down, and curl dumbbells up." },
  { name: "Reverse Curl", category: "strength", muscle_groups: ["biceps", "forearms"], equipment: "barbell", instructions: "Hold barbell with overhand grip, curl up keeping forearms engaged throughout." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — TRICEPS (10)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Overhead Tricep Extension", category: "strength", muscle_groups: ["triceps"], equipment: "dumbbell", instructions: "Hold one dumbbell overhead with both hands, lower behind your head then extend back up." },
  { name: "Tricep Pushdown", category: "strength", muscle_groups: ["triceps"], equipment: "cable", instructions: "Stand at a high cable with bar attachment, push down until arms are fully extended." },
  { name: "Rope Tricep Pushdown", category: "strength", muscle_groups: ["triceps"], equipment: "cable", instructions: "Use a rope attachment on a high cable, push down and spread the rope at the bottom." },
  { name: "Skull Crusher", category: "strength", muscle_groups: ["triceps"], equipment: "barbell", instructions: "Lie on a bench with barbell extended overhead, bend elbows to lower bar toward forehead, then extend." },
  { name: "Dumbbell Skull Crusher", category: "strength", muscle_groups: ["triceps"], equipment: "dumbbell", instructions: "Lie on a bench holding dumbbells overhead, lower toward your temples by bending elbows, then extend." },
  { name: "Tricep Dip", category: "strength", muscle_groups: ["triceps", "chest", "shoulders"], equipment: "bodyweight", instructions: "On parallel bars with upright torso, lower body by bending elbows and press back up." },
  { name: "Bench Dip", category: "strength", muscle_groups: ["triceps"], equipment: "bodyweight", instructions: "Place hands on a bench behind you, feet on the floor, lower body by bending elbows then press up." },
  { name: "Close-Grip Bench Press", category: "strength", muscle_groups: ["triceps", "chest", "shoulders"], equipment: "barbell", instructions: "Lie on a flat bench, grip barbell with hands shoulder-width apart, lower to chest and press up." },
  { name: "Tricep Kickback", category: "strength", muscle_groups: ["triceps"], equipment: "dumbbell", instructions: "Hinge forward, keep upper arm pinned to side, extend forearm back until arm is straight." },
  { name: "Overhead Cable Tricep Extension", category: "strength", muscle_groups: ["triceps"], equipment: "cable", instructions: "Face away from a low cable with rope, extend arms overhead from behind your head." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — LEGS (22)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Barbell Back Squat", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings", "core"], equipment: "barbell", instructions: "Place barbell on upper back, squat down until thighs are parallel to the floor, then stand up." },
  { name: "Barbell Front Squat", category: "strength", muscle_groups: ["quads", "glutes", "core"], equipment: "barbell", instructions: "Hold barbell across front delts in a clean grip or cross-arm position, squat to parallel." },
  { name: "Goblet Squat", category: "strength", muscle_groups: ["quads", "glutes", "core"], equipment: "dumbbell", instructions: "Hold a dumbbell vertically at chest level, squat down keeping torso upright." },
  { name: "Leg Press", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "machine", instructions: "Sit in the leg press machine, place feet shoulder-width on the platform, press until legs are extended." },
  { name: "Forward Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "bodyweight", instructions: "Step forward into a lunge until both knees are at 90 degrees, push back to standing." },
  { name: "Reverse Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "bodyweight", instructions: "Step backward into a lunge until both knees are at 90 degrees, return to standing." },
  { name: "Walking Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "bodyweight", instructions: "Perform continuous lunges walking forward, alternating legs with each step." },
  { name: "Dumbbell Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "dumbbell", instructions: "Hold dumbbells at sides, step forward into a lunge and push back to standing." },
  { name: "Leg Extension", category: "strength", muscle_groups: ["quads"], equipment: "machine", instructions: "Sit in the machine with ankles behind the pad, extend legs until straight." },
  { name: "Leg Curl", category: "strength", muscle_groups: ["hamstrings"], equipment: "machine", instructions: "Lie face down on the machine, curl heels toward glutes against the resistance." },
  { name: "Seated Leg Curl", category: "strength", muscle_groups: ["hamstrings"], equipment: "machine", instructions: "Sit in the machine with pad above ankles, curl legs under the seat." },
  { name: "Standing Calf Raise", category: "strength", muscle_groups: ["calves"], equipment: "machine", instructions: "Stand on the edge of a platform under a calf raise machine, rise onto toes then lower slowly." },
  { name: "Seated Calf Raise", category: "strength", muscle_groups: ["calves"], equipment: "machine", instructions: "Sit with knees under the pad, rise onto toes squeezing calves at the top." },
  { name: "Bodyweight Calf Raise", category: "strength", muscle_groups: ["calves"], equipment: "bodyweight", instructions: "Stand on the edge of a step, rise onto toes and slowly lower heels below the platform." },
  { name: "Barbell Hip Thrust", category: "strength", muscle_groups: ["glutes", "hamstrings"], equipment: "barbell", instructions: "Sit with upper back against a bench, barbell over hips, drive hips up until body forms a straight line." },
  { name: "Dumbbell Hip Thrust", category: "strength", muscle_groups: ["glutes", "hamstrings"], equipment: "dumbbell", instructions: "Sit with upper back against a bench, place dumbbell on hips, drive hips up squeezing glutes." },
  { name: "Romanian Deadlift", category: "strength", muscle_groups: ["hamstrings", "glutes", "back"], equipment: "barbell", instructions: "Hold barbell at hip height, hinge at hips lowering the bar along your legs while keeping back straight." },
  { name: "Dumbbell Romanian Deadlift", category: "strength", muscle_groups: ["hamstrings", "glutes"], equipment: "dumbbell", instructions: "Hold dumbbells in front of thighs, hinge at hips lowering weights while keeping a slight knee bend." },
  { name: "Bulgarian Split Squat", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "bodyweight", instructions: "Place rear foot on a bench behind you, lower into a single-leg squat on the front leg." },
  { name: "Dumbbell Bulgarian Split Squat", category: "strength", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "dumbbell", instructions: "Hold dumbbells at sides with rear foot elevated on a bench, squat down on the front leg." },
  { name: "Hack Squat", category: "strength", muscle_groups: ["quads", "glutes"], equipment: "machine", instructions: "Stand in the hack squat machine with shoulders under pads, squat down and press back up." },
  { name: "Step-Up", category: "strength", muscle_groups: ["quads", "glutes"], equipment: "dumbbell", instructions: "Hold dumbbells and step up onto a box or bench, driving through the front heel to stand." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — CORE (14)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Plank", category: "strength", muscle_groups: ["core", "shoulders"], equipment: "bodyweight", instructions: "Hold a push-up position on forearms, keeping body in a straight line from head to heels." },
  { name: "Side Plank", category: "strength", muscle_groups: ["core", "obliques"], equipment: "bodyweight", instructions: "Lie on one side propped on forearm, lift hips to form a straight line, hold." },
  { name: "Crunch", category: "strength", muscle_groups: ["core"], equipment: "bodyweight", instructions: "Lie on your back with knees bent, curl shoulders off the floor toward knees." },
  { name: "Bicycle Crunch", category: "strength", muscle_groups: ["core", "obliques"], equipment: "bodyweight", instructions: "Lie on back, alternate bringing opposite elbow to knee while extending the other leg." },
  { name: "Russian Twist", category: "strength", muscle_groups: ["core", "obliques"], equipment: "bodyweight", instructions: "Sit with knees bent and feet off the floor, rotate torso side to side touching the floor." },
  { name: "Hanging Leg Raise", category: "strength", muscle_groups: ["core", "hip flexors"], equipment: "bodyweight", instructions: "Hang from a pull-up bar, raise legs until parallel to the floor, lower with control." },
  { name: "Lying Leg Raise", category: "strength", muscle_groups: ["core", "hip flexors"], equipment: "bodyweight", instructions: "Lie flat on your back, raise legs to vertical while keeping lower back pressed to the floor." },
  { name: "Ab Wheel Rollout", category: "strength", muscle_groups: ["core", "shoulders"], equipment: "none", instructions: "Kneel holding an ab wheel, roll forward extending your body, then contract abs to roll back." },
  { name: "Cable Woodchop", category: "strength", muscle_groups: ["core", "obliques"], equipment: "cable", instructions: "Set cable high, pull diagonally across body from high to low, rotating through the torso." },
  { name: "Dead Bug", category: "strength", muscle_groups: ["core"], equipment: "bodyweight", instructions: "Lie on your back with arms and legs raised, alternately extend opposite arm and leg while keeping back flat." },
  { name: "Mountain Climber", category: "strength", muscle_groups: ["core", "hip flexors", "shoulders"], equipment: "bodyweight", instructions: "Start in push-up position, alternate driving knees toward chest at a quick pace." },
  { name: "Pallof Press", category: "strength", muscle_groups: ["core", "obliques"], equipment: "cable", instructions: "Stand perpendicular to a cable machine, press the handle straight out from your chest resisting rotation." },
  { name: "Decline Sit-Up", category: "strength", muscle_groups: ["core", "hip flexors"], equipment: "bodyweight", instructions: "Sit on a decline bench with feet locked, lower back toward the bench then sit back up." },
  { name: "Flutter Kick", category: "strength", muscle_groups: ["core", "hip flexors"], equipment: "bodyweight", instructions: "Lie on your back with legs straight, alternate small up-and-down kicks while keeping lower back pressed down." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — COMPOUND / FULL BODY (8)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Barbell Row to Press", category: "strength", muscle_groups: ["back", "shoulders", "biceps", "triceps"], equipment: "barbell", instructions: "Perform a bent-over row, stand up, then press the barbell overhead in one fluid motion." },
  { name: "Turkish Get-Up", category: "strength", muscle_groups: ["shoulders", "core", "glutes", "quads"], equipment: "kettlebell", instructions: "Lie on your back holding a kettlebell overhead, stand up in stages while keeping the weight locked out." },
  { name: "Farmer's Walk", category: "strength", muscle_groups: ["forearms", "traps", "core"], equipment: "dumbbell", instructions: "Hold heavy dumbbells at your sides and walk with upright posture for distance or time." },
  { name: "Trap Bar Deadlift", category: "strength", muscle_groups: ["back", "quads", "glutes", "hamstrings"], equipment: "barbell", instructions: "Stand inside a trap bar, grip the handles, and stand up by extending hips and knees." },
  { name: "Dumbbell Pullover", category: "strength", muscle_groups: ["chest", "back"], equipment: "dumbbell", instructions: "Lie on a bench holding one dumbbell overhead, lower it behind your head in an arc, then pull back over chest." },
  { name: "Barbell Good Morning", category: "strength", muscle_groups: ["hamstrings", "back", "glutes"], equipment: "barbell", instructions: "Place barbell on upper back, hinge at hips with slight knee bend until torso is nearly parallel to floor." },
  { name: "Glute Bridge", category: "strength", muscle_groups: ["glutes", "hamstrings"], equipment: "bodyweight", instructions: "Lie on your back with knees bent, drive hips upward squeezing glutes at the top." },
  { name: "Single-Leg Deadlift", category: "strength", muscle_groups: ["hamstrings", "glutes", "core"], equipment: "dumbbell", instructions: "Stand on one leg, hinge at the hip lowering the dumbbell while extending the free leg behind you." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — FOREARMS & GRIP (4)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Wrist Curl", category: "strength", muscle_groups: ["forearms"], equipment: "barbell", instructions: "Sit with forearms on thighs, palms up, curl the barbell up using only your wrists." },
  { name: "Reverse Wrist Curl", category: "strength", muscle_groups: ["forearms"], equipment: "barbell", instructions: "Sit with forearms on thighs, palms down, extend wrists upward against resistance." },
  { name: "Plate Pinch Hold", category: "strength", muscle_groups: ["forearms"], equipment: "none", instructions: "Pinch two weight plates together smooth-side-out, hold for time to build grip strength." },
  { name: "Dead Hang", category: "strength", muscle_groups: ["forearms", "back", "shoulders"], equipment: "bodyweight", instructions: "Hang from a pull-up bar with straight arms for as long as possible to build grip endurance." },

  // ═══════════════════════════════════════════════════════════════════════════
  // STRENGTH — BAND EXERCISES (6)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Band Pull-Apart", category: "strength", muscle_groups: ["rear delts", "traps", "rotator cuff"], equipment: "band", instructions: "Hold a resistance band at shoulder width, pull it apart by squeezing shoulder blades together." },
  { name: "Band Bicep Curl", category: "strength", muscle_groups: ["biceps"], equipment: "band", instructions: "Stand on the band, grip the other end and curl up against the resistance." },
  { name: "Band Lateral Walk", category: "strength", muscle_groups: ["glutes", "hip abductors"], equipment: "band", instructions: "Place a band around ankles or above knees, take side steps maintaining tension on the band." },
  { name: "Band Face Pull", category: "strength", muscle_groups: ["rear delts", "rotator cuff"], equipment: "band", instructions: "Anchor band at face height, pull toward your face with elbows high and externally rotate at the end." },
  { name: "Band Tricep Pushdown", category: "strength", muscle_groups: ["triceps"], equipment: "band", instructions: "Anchor band overhead, grip with both hands and push down until arms are fully extended." },
  { name: "Band Squat", category: "strength", muscle_groups: ["quads", "glutes"], equipment: "band", instructions: "Stand on the band with feet shoulder-width, hold at shoulders, and squat down against the resistance." },

  // ═══════════════════════════════════════════════════════════════════════════
  // CARDIO (32)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Treadmill Running", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves", "core"], equipment: "cardio_machine", instructions: "Run on a treadmill at a steady or interval pace, adjusting speed and incline to target heart rate." },
  { name: "Outdoor Running", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves", "core"], equipment: "none", instructions: "Run outdoors at a steady pace or with intervals, maintaining good posture and foot strike." },
  { name: "Sprints", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves", "glutes"], equipment: "none", instructions: "Run at maximum effort for 20-60 seconds, rest, and repeat for the desired number of intervals." },
  { name: "Hill Sprints", category: "cardio", muscle_groups: ["quads", "glutes", "calves", "hamstrings"], equipment: "none", instructions: "Sprint up a steep hill at max effort, walk back down to recover, and repeat." },
  { name: "Stationary Cycling", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves"], equipment: "cardio_machine", instructions: "Pedal on a stationary bike at moderate to high intensity, adjusting resistance as needed." },
  { name: "Outdoor Cycling", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves"], equipment: "none", instructions: "Cycle outdoors on roads or trails, maintaining a steady cadence and adjusting gears for terrain." },
  { name: "Rowing Machine", category: "cardio", muscle_groups: ["back", "legs", "arms", "core"], equipment: "cardio_machine", instructions: "Use a rowing machine with proper form: drive with legs, lean back slightly, then pull handle to chest." },
  { name: "Swimming", category: "cardio", muscle_groups: ["back", "shoulders", "core", "legs"], equipment: "none", instructions: "Swim laps using freestyle, backstroke, or other strokes, maintaining consistent breathing rhythm." },
  { name: "Jump Rope", category: "cardio", muscle_groups: ["calves", "shoulders", "core"], equipment: "none", instructions: "Jump over a spinning rope with both feet, keeping bounces low and wrists loose." },
  { name: "Double-Under Jump Rope", category: "cardio", muscle_groups: ["calves", "shoulders", "core"], equipment: "none", instructions: "Swing the rope twice under your feet per jump, requiring faster wrist rotation and higher jumps." },
  { name: "Elliptical Trainer", category: "cardio", muscle_groups: ["quads", "hamstrings", "glutes"], equipment: "cardio_machine", instructions: "Use the elliptical with a smooth stride, pushing and pulling handles for full-body engagement." },
  { name: "Stair Climber", category: "cardio", muscle_groups: ["quads", "glutes", "calves"], equipment: "cardio_machine", instructions: "Step continuously on the stair climber machine at a steady pace, driving through each step." },
  { name: "Stair Running", category: "cardio", muscle_groups: ["quads", "glutes", "calves", "core"], equipment: "none", instructions: "Run up a flight of stairs quickly, walk or jog back down, and repeat." },
  { name: "Battle Ropes", category: "cardio", muscle_groups: ["shoulders", "arms", "core"], equipment: "none", instructions: "Grip heavy ropes and create alternating or simultaneous waves by rapidly raising and lowering your arms." },
  { name: "Box Jump", category: "cardio", muscle_groups: ["quads", "glutes", "calves"], equipment: "none", instructions: "Stand facing a sturdy box, jump onto it landing softly with both feet, step back down." },
  { name: "Burpee", category: "cardio", muscle_groups: ["chest", "quads", "core", "shoulders"], equipment: "bodyweight", instructions: "Drop to a push-up, perform the push-up, jump feet to hands, then explosively jump up with arms overhead." },
  { name: "Jumping Jack", category: "cardio", muscle_groups: ["calves", "shoulders", "core"], equipment: "bodyweight", instructions: "Jump feet apart while raising arms overhead, then jump back together." },
  { name: "High Knees", category: "cardio", muscle_groups: ["quads", "hip flexors", "calves"], equipment: "bodyweight", instructions: "Run in place driving knees as high as possible with each step at a fast pace." },
  { name: "Butt Kicks", category: "cardio", muscle_groups: ["hamstrings", "calves"], equipment: "bodyweight", instructions: "Run in place kicking heels up to touch your glutes with each step." },
  { name: "Walking", category: "cardio", muscle_groups: ["quads", "hamstrings", "calves"], equipment: "none", instructions: "Walk at a brisk pace, swinging arms naturally, for sustained cardiovascular benefit." },
  { name: "Incline Walking", category: "cardio", muscle_groups: ["quads", "glutes", "calves"], equipment: "cardio_machine", instructions: "Walk on a treadmill at a steep incline (10-15%) at a moderate pace for sustained effort." },
  { name: "Assault Bike", category: "cardio", muscle_groups: ["quads", "hamstrings", "shoulders", "core"], equipment: "cardio_machine", instructions: "Pedal and push/pull the handles simultaneously on an air resistance bike for max calorie burn." },
  { name: "Ski Erg", category: "cardio", muscle_groups: ["back", "shoulders", "core", "triceps"], equipment: "cardio_machine", instructions: "Stand at the ski erg, pull both handles down simultaneously while hinging at the hips." },
  { name: "Bear Crawl", category: "cardio", muscle_groups: ["shoulders", "core", "quads"], equipment: "bodyweight", instructions: "Get on all fours with knees hovering, crawl forward by moving opposite hand and foot together." },
  { name: "Shadow Boxing", category: "cardio", muscle_groups: ["shoulders", "core", "arms"], equipment: "bodyweight", instructions: "Throw punches in the air with proper form while bouncing on your feet for cardio conditioning." },
  { name: "Rowing Sprint", category: "cardio", muscle_groups: ["back", "legs", "arms", "core"], equipment: "cardio_machine", instructions: "Row at maximum intensity for 250-500 meters, rest briefly, and repeat." },
  { name: "Swimming Sprints", category: "cardio", muscle_groups: ["back", "shoulders", "core", "legs"], equipment: "none", instructions: "Swim one lap at maximum speed, rest at the wall, and repeat for the target number of sets." },
  { name: "Sled Push", category: "cardio", muscle_groups: ["quads", "glutes", "core", "shoulders"], equipment: "none", instructions: "Load a sled and push it across the floor by driving through your legs with arms extended." },
  { name: "Sled Pull", category: "cardio", muscle_groups: ["back", "hamstrings", "biceps"], equipment: "none", instructions: "Attach a rope to a sled, walk backward or pull hand-over-hand to drag the sled toward you." },
  { name: "Tuck Jump", category: "cardio", muscle_groups: ["quads", "glutes", "calves", "core"], equipment: "bodyweight", instructions: "Jump explosively, tucking knees to chest at the peak, and land softly." },
  { name: "Skater Jump", category: "cardio", muscle_groups: ["quads", "glutes", "hip abductors"], equipment: "bodyweight", instructions: "Jump laterally from one foot to the other, landing softly and reaching the opposite hand to the ground." },
  { name: "Broad Jump", category: "cardio", muscle_groups: ["quads", "glutes", "hamstrings"], equipment: "bodyweight", instructions: "Stand with feet hip-width, swing arms and jump forward as far as possible, landing softly on both feet." },

  // ═══════════════════════════════════════════════════════════════════════════
  // FLEXIBILITY (22)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Downward Dog", category: "flexibility", muscle_groups: ["hamstrings", "calves", "shoulders", "back"], equipment: "none", instructions: "From all fours, lift hips up and back forming an inverted V, pressing heels toward the floor." },
  { name: "Warrior I", category: "flexibility", muscle_groups: ["hip flexors", "quads", "shoulders"], equipment: "none", instructions: "Step one foot forward into a lunge, raise arms overhead, hips square to the front." },
  { name: "Warrior II", category: "flexibility", muscle_groups: ["hip flexors", "quads", "groin"], equipment: "none", instructions: "From a wide stance, bend the front knee, extend arms parallel to the floor, gaze over the front hand." },
  { name: "Warrior III", category: "flexibility", muscle_groups: ["hamstrings", "glutes", "core"], equipment: "none", instructions: "Balance on one leg, extend the other leg behind and arms forward, forming a T-shape with your body." },
  { name: "Child's Pose", category: "flexibility", muscle_groups: ["back", "hips", "shoulders"], equipment: "none", instructions: "Kneel and sit back on heels, extend arms forward on the floor, resting forehead down." },
  { name: "Cobra Pose", category: "flexibility", muscle_groups: ["core", "back", "hip flexors"], equipment: "none", instructions: "Lie face down, place hands under shoulders, press up lifting chest while keeping hips on the floor." },
  { name: "Pigeon Pose", category: "flexibility", muscle_groups: ["glutes", "hip flexors", "piriformis"], equipment: "none", instructions: "From all fours, bring one knee forward behind the same wrist, extend the other leg back, sink hips down." },
  { name: "Standing Hamstring Stretch", category: "flexibility", muscle_groups: ["hamstrings"], equipment: "none", instructions: "Stand and place one heel on an elevated surface, keeping the leg straight, hinge forward from the hips." },
  { name: "Seated Hamstring Stretch", category: "flexibility", muscle_groups: ["hamstrings", "back"], equipment: "none", instructions: "Sit with legs extended, reach forward toward your toes keeping your back as straight as possible." },
  { name: "Standing Quad Stretch", category: "flexibility", muscle_groups: ["quads", "hip flexors"], equipment: "none", instructions: "Stand on one leg, pull the other foot toward your glute, keeping knees together." },
  { name: "Kneeling Hip Flexor Stretch", category: "flexibility", muscle_groups: ["hip flexors", "quads"], equipment: "none", instructions: "Kneel on one knee, push hips forward while keeping torso upright to stretch the front of the hip." },
  { name: "Chest Doorway Stretch", category: "flexibility", muscle_groups: ["chest", "shoulders"], equipment: "none", instructions: "Stand in a doorway with arms on the frame at 90 degrees, step forward to stretch the chest and front shoulders." },
  { name: "Cross-Body Shoulder Stretch", category: "flexibility", muscle_groups: ["shoulders", "rear delts"], equipment: "none", instructions: "Pull one arm across your body at chest height using the opposite hand." },
  { name: "Overhead Tricep Stretch", category: "flexibility", muscle_groups: ["triceps", "shoulders"], equipment: "none", instructions: "Reach one arm overhead, bend the elbow, and use the other hand to gently pull the elbow behind your head." },
  { name: "Standing Calf Stretch", category: "flexibility", muscle_groups: ["calves"], equipment: "none", instructions: "Place hands on a wall, step one foot back keeping it straight, press the heel into the floor." },
  { name: "IT Band Stretch", category: "flexibility", muscle_groups: ["it band", "hips"], equipment: "none", instructions: "Stand with legs crossed, reach the arm of the back leg overhead and lean to the side." },
  { name: "Figure-Four Stretch", category: "flexibility", muscle_groups: ["glutes", "piriformis", "hips"], equipment: "none", instructions: "Lie on your back, cross one ankle over the opposite knee, pull the bottom leg toward your chest." },
  { name: "Cat-Cow Stretch", category: "flexibility", muscle_groups: ["back", "core"], equipment: "none", instructions: "On all fours, alternate between arching your back (cow) and rounding your spine (cat)." },
  { name: "Thread the Needle", category: "flexibility", muscle_groups: ["shoulders", "upper back"], equipment: "none", instructions: "On all fours, slide one arm under the other across the floor, rotating your upper back." },
  { name: "Seated Spinal Twist", category: "flexibility", muscle_groups: ["back", "obliques", "hips"], equipment: "none", instructions: "Sit with one leg extended, cross the other foot over, twist your torso toward the bent knee." },
  { name: "Butterfly Stretch", category: "flexibility", muscle_groups: ["groin", "hips"], equipment: "none", instructions: "Sit with soles of feet together, gently press knees toward the floor with elbows." },
  { name: "Supine Spinal Twist", category: "flexibility", muscle_groups: ["back", "glutes", "obliques"], equipment: "none", instructions: "Lie on your back, pull one knee across your body to the opposite side while keeping shoulders flat." },

  // ═══════════════════════════════════════════════════════════════════════════
  // HIIT (22)
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Kettlebell Swing", category: "hiit", muscle_groups: ["glutes", "hamstrings", "core", "shoulders"], equipment: "kettlebell", instructions: "Hinge at hips, swing kettlebell between legs, then explosively drive hips forward to swing it to chest height." },
  { name: "Single-Arm Kettlebell Swing", category: "hiit", muscle_groups: ["glutes", "hamstrings", "core", "shoulders"], equipment: "kettlebell", instructions: "Perform a standard kettlebell swing using one hand, switching hands each set." },
  { name: "Barbell Thruster", category: "hiit", muscle_groups: ["quads", "glutes", "shoulders", "triceps"], equipment: "barbell", instructions: "Front squat with a barbell, then drive up explosively pressing the bar overhead in one movement." },
  { name: "Dumbbell Thruster", category: "hiit", muscle_groups: ["quads", "glutes", "shoulders", "triceps"], equipment: "dumbbell", instructions: "Hold dumbbells at shoulders, squat down, then drive up pressing the weights overhead." },
  { name: "Wall Ball", category: "hiit", muscle_groups: ["quads", "glutes", "shoulders", "core"], equipment: "none", instructions: "Hold a medicine ball at chest, squat, then explosively stand and throw the ball to a target on the wall." },
  { name: "Medicine Ball Slam", category: "hiit", muscle_groups: ["core", "shoulders", "back"], equipment: "none", instructions: "Lift a medicine ball overhead, then slam it to the ground as hard as possible, catch and repeat." },
  { name: "Lateral Medicine Ball Slam", category: "hiit", muscle_groups: ["core", "obliques", "shoulders"], equipment: "none", instructions: "Hold a medicine ball overhead, rotate and slam it to one side, alternate sides." },
  { name: "Barbell Clean and Jerk", category: "hiit", muscle_groups: ["quads", "glutes", "shoulders", "back", "core"], equipment: "barbell", instructions: "Pull barbell from floor to shoulders in one explosive motion (clean), then press or jerk overhead." },
  { name: "Barbell Snatch", category: "hiit", muscle_groups: ["quads", "glutes", "shoulders", "back", "core"], equipment: "barbell", instructions: "Pull barbell from floor to overhead in one explosive motion with a wide grip." },
  { name: "Power Clean", category: "hiit", muscle_groups: ["quads", "glutes", "hamstrings", "back", "shoulders"], equipment: "barbell", instructions: "Pull barbell explosively from the floor, catch it at shoulder height in a partial squat position." },
  { name: "Kettlebell Clean and Press", category: "hiit", muscle_groups: ["shoulders", "core", "glutes"], equipment: "kettlebell", instructions: "Clean a kettlebell to the rack position, then press overhead, lower and repeat." },
  { name: "Kettlebell Snatch", category: "hiit", muscle_groups: ["shoulders", "back", "glutes", "core"], equipment: "kettlebell", instructions: "Swing a kettlebell between legs, then pull it overhead in one fluid motion, punching hand through at the top." },
  { name: "Tire Flip", category: "hiit", muscle_groups: ["back", "glutes", "quads", "shoulders", "core"], equipment: "none", instructions: "Squat down gripping under a large tire, drive up explosively flipping the tire over." },
  { name: "Battle Rope Slam", category: "hiit", muscle_groups: ["shoulders", "arms", "core"], equipment: "none", instructions: "Grip heavy ropes and slam them to the ground simultaneously with maximum force." },
  { name: "HIIT Box Jump", category: "hiit", muscle_groups: ["quads", "glutes", "calves"], equipment: "none", instructions: "Perform rapid box jumps with minimal ground contact time, stepping or jumping back down between reps." },
  { name: "Kettlebell Goblet Squat", category: "hiit", muscle_groups: ["quads", "glutes", "core"], equipment: "kettlebell", instructions: "Hold a kettlebell at chest level by the horns, squat deep keeping torso upright." },
  { name: "Devil Press", category: "hiit", muscle_groups: ["chest", "shoulders", "glutes", "core"], equipment: "dumbbell", instructions: "Perform a burpee with dumbbells, then swing or snatch both dumbbells overhead as you stand." },
  { name: "Renegade Row", category: "hiit", muscle_groups: ["back", "core", "shoulders"], equipment: "dumbbell", instructions: "In push-up position holding dumbbells, alternate rowing each weight to your hip while stabilizing." },
  { name: "Man Maker", category: "hiit", muscle_groups: ["chest", "back", "shoulders", "quads", "core"], equipment: "dumbbell", instructions: "Perform a push-up on dumbbells, row each side, jump feet to hands, clean and press overhead." },
  { name: "Hang Clean", category: "hiit", muscle_groups: ["quads", "glutes", "back", "shoulders"], equipment: "barbell", instructions: "Start with barbell at hip height, explosively pull to shoulders catching in a partial squat." },
  { name: "Push Press", category: "hiit", muscle_groups: ["shoulders", "triceps", "quads"], equipment: "barbell", instructions: "Dip slightly at the knees, then explosively drive the barbell overhead using leg power." },
  { name: "Kettlebell Turkish Get-Up", category: "hiit", muscle_groups: ["shoulders", "core", "glutes", "quads"], equipment: "kettlebell", instructions: "Lie on your back holding a kettlebell overhead, stand up in stages keeping the weight locked out, then reverse." },

  // ═══════════════════════════════════════════════════════════════════════════
  // ADDITIONAL — push past 200
  // ═══════════════════════════════════════════════════════════════════════════
  { name: "Dumbbell Lateral Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hip adductors"], equipment: "dumbbell", instructions: "Hold dumbbells at sides, step wide to one side bending that knee while keeping the other leg straight." },
  { name: "Cable Pull-Through", category: "strength", muscle_groups: ["glutes", "hamstrings"], equipment: "cable", instructions: "Face away from a low cable, hinge at the hips, then drive hips forward pulling the cable between your legs." },
  { name: "Sissy Squat", category: "strength", muscle_groups: ["quads"], equipment: "bodyweight", instructions: "Stand on toes, lean back bending only at the knees while keeping hips extended, then return." },
  { name: "Sumo Squat", category: "strength", muscle_groups: ["quads", "glutes", "hip adductors"], equipment: "dumbbell", instructions: "Take a wide stance with toes pointed out, hold a dumbbell between legs and squat deep." },
  { name: "Copenhagen Plank", category: "strength", muscle_groups: ["core", "hip adductors"], equipment: "bodyweight", instructions: "Side plank with top foot on a bench and bottom leg hanging, lift hips and hold." },
  { name: "V-Up", category: "strength", muscle_groups: ["core", "hip flexors"], equipment: "bodyweight", instructions: "Lie flat, simultaneously raise legs and torso reaching hands toward toes forming a V shape." },
  { name: "Hollow Body Hold", category: "strength", muscle_groups: ["core"], equipment: "bodyweight", instructions: "Lie on your back, raise arms overhead and legs off the floor, press lower back flat, and hold." },
  { name: "Curtsy Lunge", category: "strength", muscle_groups: ["quads", "glutes", "hip adductors"], equipment: "bodyweight", instructions: "Step one foot behind and across the other leg, lower into a lunge, then return to standing." },
];

// ─────────────────────────────────────────────────────────────────────────────
// Seed logic
// ─────────────────────────────────────────────────────────────────────────────

async function seed() {
  const sql = postgres(DATABASE_URL, { prepare: false });

  console.log(`Seeding ${exercises.length} exercises...`);

  // Ensure a unique constraint on name exists for ON CONFLICT
  // (If it doesn't exist, create it — idempotent)
  await sql`
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'exercises_name_key'
      ) THEN
        ALTER TABLE exercises ADD CONSTRAINT exercises_name_key UNIQUE (name);
      END IF;
    END $$;
  `;

  // Insert all exercises in a single transaction
  let inserted = 0;

  await sql.begin(async (tx) => {
    for (const e of exercises) {
      await tx`
        INSERT INTO exercises (name, category, muscle_groups, equipment, instructions, is_custom, created_by)
        VALUES (${e.name}, ${e.category}, ${sql.array(e.muscle_groups)}::text[], ${e.equipment}, ${e.instructions}, false, NULL)
        ON CONFLICT (name) DO NOTHING
      `;
      inserted++;
    }
  });

  console.log(`Inserted ${inserted} exercises (skipping any duplicates).`);

  // Verify count
  const [{ count }] = await sql`SELECT count(*)::int as count FROM exercises WHERE is_custom = false`;
  console.log(`\nDone. Total library exercises in DB: ${count}`);

  await sql.end();
}

seed().catch((err) => {
  console.error("Seed failed:", err);
  process.exit(1);
});
