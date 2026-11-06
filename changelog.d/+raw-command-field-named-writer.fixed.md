A raw whose `Command` field named its writer inside a longer word, such as
`QSPICE64`, was refused as an ambiguous dialect when read by path, while
the progress of a running job named it. Both now read the field by one
rule: a simulator's name opening a word, with a build number or word after
it allowed.
