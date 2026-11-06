An argument refusal names the fix for a field it does not know: an unknown
field is answered with the names its place takes, the closest first ("Did
you mean 'dry_run' for 'dry_runn'? This tool takes path, base, ..."), where
it said only that the field was not permitted. A missing field that chooses
a branch is answered with the values it takes (`jobs` without `action`,
an `edit_schematic` op without `op`), where pydantic's "Unable to extract
tag using discriminator" named neither. The item's fields that no branch
takes are refused beside it, so a misspelled `action` or `op` is named
(validation stopped at the missing field and never reached them).
