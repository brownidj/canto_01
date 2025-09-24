- When more than one syllable, should they be split up to show different tones?
- Prosodic (LSHK) (one word Jyutping) or Pedagogical? Add to setup. pycantonese should always output learner-friendlyJyutping
- Do we want to show word boundaries for strict, eg for 廣播站 can be shown as gwong2 bo3 · zaam6 or gwong2bo3 | zaam6
- Here are some examples: 

**1 format_jyutping("飲嘢", style="learner")**

"jam2 je5"

**2 format_jyutping("飲嘢", style="strict")**

"jam2je5"

**3 format_jyutping("廣播站", style="learner")**

"gwong2 bo3 zaam6"

**4 format_jyutping("廣播站", style="strict")**

"gwong2bo3 zaam6"

**5 format_jyutping("廣播站", style="strict_with_word_boundaries", word_boundary_marker=" · ")**

"gwong2bo3 · zaam6"