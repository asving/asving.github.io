# Scientific Writing Skills

## I. McCarthy's rules (from Cormac McCarthy, via Savage & Yeh)

## Core principle
Use minimalism to achieve clarity. Remove everything that does not serve the reader.

## Sentence-level rules
- **Short sentences.** Each sentence should contain one idea. If a sentence has more than one clause, ask whether it should be two sentences.
- **No semicolons.** They are unnecessary. Use a period or restructure.
- **Avoid "which" in restrictive clauses.** Use "that" instead. ("The result that we obtained" not "the result which we obtained.")
- **Avoid "while" meaning "whereas" or "although."** Reserve "while" for simultaneity.
- **Don't use "in order to."** Just write "to."
- **Don't use "respectively."** Rewrite the sentence instead.
- **Minimize commas.** If a sentence needs many commas, it is too complex.
- **Use "but" rather than "however."**
- **Avoid parenthetical asides.** If it matters, give it its own sentence. If it doesn't, cut it.

## Word-level rules
- **Cut modifiers.** Remove "very," "really," "quite," "rather," "somewhat," "significantly" (when not statistical).
- **Prefer active voice.** "We measured X" not "X was measured."
- **Avoid jargon** when a plain word exists.
- **Use concrete language** over abstract language.
- **Don't hedge excessively.** Say what you mean.

## Paragraph and structure
- **Keep paragraphs short.** Three to five sentences. A wall of text loses the reader.
- **The first sentence of each paragraph should state the point.** The rest supports it.
- **Read your text aloud.** If you stumble, rewrite.

## General
- **Avoid footnotes** in the main argument.
- **Minimize use of "and" in lists.** If you have a long list joined by "and," rethink the structure.
- **Every word must earn its place.** If removing a word does not change the meaning, remove it.

## II. Argument and structure

- **One argument per essay.** Every section should serve a single throughline. If a section does not advance the main argument, cut it.
- **Don't say the same thing twice.** If two sections make the same point, merge them. Redundancy signals unclear thinking about what the point actually is.
- **The reader should never wonder "why am I reading this."** At the start of each section, make clear what it will argue and why it matters to the main thread.
- **Transitions should feel inevitable.** The end of one section should make the reader want to read the next. Not through clunky signposts ("In this section we will...") but through logical momentum.

## III. Concreteness

- **Example first, then principle.** Readers anchor to concrete cases and generalize from them. State the general principle after the reader has seen what it looks like. This is how understanding works.
- **Analogies should make the unfamiliar familiar.** Not the familiar sound profound. If an analogy requires as much explanation as the thing it illustrates, cut it.
- **Precision over impressiveness.** "This is hard" beats "this presents formidable challenges." "We don't know how to do this" beats "this remains an open frontier." Say exactly what you mean.

## IV. Respect the reader

- **Don't explain what the audience knows.** For a math audience: don't explain what a proof is. For ML researchers: don't explain what a neural network is. Spend words on what is new.
- **Don't use rhetorical questions as a crutch.** "Can we mechanize insight?" is weaker than "The question is whether insight can be mechanized." State the question, don't perform it.
- **Avoid the TED-talk register.** No "This is not idle speculation." No "The answer may surprise you." Trust the content to carry the weight.

## V. Tension and stakes

- **Good writing has a problem driving it forward.** The reader should feel: here is what we want to understand, here is why it is hard, here is what we can say. Not everything resolved, but the difficulty made vivid.
- **Be honest about what you don't know.** Admitting limits is more persuasive than hedging. "We do not know how to do X" is stronger than "X remains a challenging open question for future work."
- **Cut the victory lap.** Don't celebrate your own framework. Present it and let the reader judge.

## VI. Openings and voice

- **The opening earns the reader's attention.** The first sentence should raise a question or state something precise enough to be interesting. Not a throat-clearing preamble ("Throughout history, humans have...") but an entry point that makes the reader lean in. Darwin opens *On the Origin of Species* with pigeons. Concrete, specific, immediately engaging.
- **Earn your abstractions.** Every abstract claim should be preceded by the concrete observation that forced you to it. "LLMs are part of collective consciousness" is an assertion. Showing the specific interaction that made this framing feel necessary is an argument.
- **Have a position.** The best essays don't survey. They argue. The reader should be able to disagree with you. If nobody could disagree, you haven't said anything.
- **Mark the boundary between knowledge and speculation.** Not hedging. Say "here is what I can show, and here is what I suspect but cannot prove." The reader trusts you because you mark the boundary clearly.

## VII. Process

- **Write to discover, then rewrite to present.** First drafts figure out what you think. Final drafts show the reader what you found. These are different activities. The best essays preserve the feeling of thought-in-progress but the writer already knows where it's going.

## VIII. Avoiding LLM-ese (when co-writing with an LLM)

LLM prose has a recognizable voice. When co-writing, audit for these tells:

### Words and phrases to cut
- **Intensifiers that add gravity without content.** "Genuinely," "profoundly," "fundamentally," "without precedent," "in a meaningful sense." If the claim is strong, it doesn't need to announce itself. Just state it.
- **Signpost phrases.** "It is worth pausing to consider," "This brings up what I think is the most important," "This matters directly for." These perform the act of thinking rather than doing it. Just present the thing.
- **Hedging tics.** Excessive "I think" used as a softener rather than to mark a genuine intellectual position. "Something like" and "or perhaps" when you can just commit to the word.
- **Vague gravitas.** "The nature of what we are dealing with," "the thing we have built," "what it is we have done." Say what it is specifically.

### Structural tells
- **Rhythmic parallelism.** Three or more sentences with identical structure in a row ("X does A but cannot B. Y does C but cannot D. Z does E but cannot F.") is a reliable LLM signature. Vary the structure. Compress some into a list, expand others, break the rhythm.
- **Even paragraph length.** LLMs default to paragraphs of roughly equal size, each with a topic sentence, development, and clean close. Vary paragraph length. Some paragraphs should be two sentences. Others can be dense.
- **Triads.** LLMs love three-part constructions ("personalized, universal, and rapidly recursive"; "profoundly and pervasively"). Use them sparingly. One triad per essay is fine; five is a pattern.
- **The sweeping close.** A long building sentence with parallel clauses that crescendos into a final image. End essays with concrete questions, abrupt stops, or specific images instead.

### Voice calibration
- **Compare against the author's other writing.** The strongest test. If the author writes with specific technical examples, varied paragraph rhythm, and sparse first person, the co-written piece should too.
- **First person should be functional.** "I will argue," "I propose" — marking the direction of the argument. Not "I think" repeated as a softener on every other claim.
- **Bold claims need no decoration.** "This is the central claim:" followed by the claim. Not "This is, I think, the most important and perhaps most underappreciated aspect of what we are seeing."
