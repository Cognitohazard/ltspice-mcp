`analyze_results` keeps an entry in `results` for every recipe
the page reached, in request order. A recipe that failed on every run, was
invalid or read a missing file lost its key, so `result["results"][key]`
raised `KeyError` for some causes only. Such a recipe now has no rows and,
when it failed, a `No rows:` warning naming the failures. The attached
analysis and the Python API give the same entries.
