An error raised under a narrower type of a hinted family reached the caller
with no recovery hint, and one reported inside a reply (a coverage row, a
batch item, a `jobs` error) was hinted only by its exact type. Both now
take the nearest hint in the error's family.
