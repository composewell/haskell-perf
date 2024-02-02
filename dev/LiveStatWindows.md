# Windows

In a linear flow, we have the ability to start a window anywhere and
stop it anywhere. A winow is basically the realestate between any 2
points in the linear code.

The goal is to make it easier and convinent to create windows.  A
point in the linear flow should have some identification. We can use
the `ModuleName:LineNumber` to identify this. Using this has two
advantages. One being, removal of developer overhead of thinking about
a name and the other being extremely fast identification of whether
the point is enabled or not based on the line number.

A window can be indentified by a tag. The tag, can be derived from
point identification. We can use
`ModuleName:LineNumber-ModuleName:LineNumber` respectively.

This nomenclature lets us create non overlapping windows and
overlapping windows windows that are completely encapsulated.

Non overlapping:
```
............[...........].....................................
......................................[.............].........
```

Overlapping:
```
............[.......................................].........
.....................[................].......................
```


Window like this aren't possible:
```
............[.........................].......................
.....................[...........................]............
```

From my experience, These kind of windows are rarely needed. But for
the 1% use-cases we can tackle this problem.

To make the windows like the above possible, we can use namespaces.
Windows can belong to certain namespaces.

We can use the `NameSpace[ModuleName:LineNumber]` to identify a point,
and `NameSpace[ModuleName:LineNumber-ModuleName:LineNumber]` to
identify a window.
