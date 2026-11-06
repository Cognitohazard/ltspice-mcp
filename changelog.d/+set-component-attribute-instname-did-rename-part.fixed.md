`set_component_attribute` with `InstName` did not rename the part. It wrote
the old name and the new one into the file, kept the part under the old
name, and reported that the part had no name before. It now renames the
part, and refuses a name that another part has in any case.
