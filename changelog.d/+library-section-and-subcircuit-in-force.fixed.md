A `.lib file section` now leaves out the cards of the library's other
sections everywhere, not only in hierarchy discovery: a model or a `.save`
that only another corner of the library declares no longer counts as present
for the `model-missing` lint rule or for the analyses and saves a run expects
from the deck.
