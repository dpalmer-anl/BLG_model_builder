import flatgraphene as fg
print("ok", getattr(fg, "__file__", fg))
atoms = fg.shift.make_graphene(
    ["A", "B"], "hex", 1, 1, 2.469, 2, 7.0,
    sym=["C", "C"], mass=[12.01, 12.01], mol_id=None, h_vac=30,
)
print("cell", atoms.cell.array)
print("pos", atoms.positions)
print("nat", len(atoms))
