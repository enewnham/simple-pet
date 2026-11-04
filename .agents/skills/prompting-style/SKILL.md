---
name: prompting-style
description: Apply Simple Pet's principles for model reasoning, information selection, factual grounding, and persona when designing or reviewing model inputs and outputs.
---

# Prompting style

Simple Pet uses a lightweight model to turn an understood situation into grounded,
personal dialogue. Shape the information so that little interpretive work remains
by the time the model expresses it. These principles guide design choices. Let the
implementation establish the appropriate components and representations.

## Spend reasoning deliberately

Use model reasoning where language judgment adds value. Resolve predictable
calculations, relationships, and policy decisions before generation. A model
should receive the conclusions the application can establish reliably without
having to reconstruct them from raw observations.

Some tasks require interpretation, including summarizing intent from prose.
Give that interpretation a clear purpose and preserve its uncertainty. Keep each
model task focused on the judgment it needs to make.

## Design the information we supply

Select, summarize, and format information for the particular response being
requested. Every additional fact creates another possible subject or
interpretation. Include it because it helps the model make a necessary choice.

Relevance comes before size limits. A short prompt can still contain distracting
or ambiguous information. Judge the complete input by what it asks the model to
notice, connect, and decide. Choose detail and precision in proportion to their
meaning for the response.

## Establish meaning before voice

Establish what happened, what matters, and the communicative purpose before
asking for persona expression. Persona shapes the wording, rhythm, emphasis,
and emotional character of that meaning.

Apply the parts of persona that can be decided beforehand when framing the
response's purpose. Leave room for natural expression without making the model
invent facts, motives, or urgency to sound in character. The intended result is
varied language grounded in a clear situation.

## Preserve truth through compression

Carry scope, uncertainty, and the distinction between intent and outcome through
selection and summarization. Condensed evidence must support the same claims as
its source. Keep observations and interpretations distinguishable.

Factual memory should preserve what can be said about the work. Persona belongs
to its expression. Retain enough grounding to judge whether a generated claim is
supported, even when the model receives only a small part of the available
history.

## Treat failures as information-design feedback

When the model repeatedly misreads a situation, inspect what it received and
which decisions were left for it. Unnecessary detail, ambiguous framing, and
competing instructions can create reasoning work that clearer input would avoid.
Address the cause before accumulating corrective instructions.

Evaluate whether the model understood the supplied meaning, preserved factual
scope, and expressed the intended persona. Use actual generated responses to
support claims about those qualities. Increase model capability or input volume
when the task demonstrates a need for it.
