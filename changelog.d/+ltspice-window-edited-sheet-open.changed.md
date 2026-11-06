When an LTspice window that has an edited sheet open does not take the
committed sheet, the server now closes the sheet there and opens it again
from the file, and the window's entry in `open_in_ltspice` says
`reopened: true`. Before, the reply told the user to do this by hand. It is
done only when the window still holds the sheet as it was before the edit,
so nothing the user changed there since is lost; otherwise the entry says
`shown: false` as before.
