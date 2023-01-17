import firedrake as fd

mesh = fd.UnitIntervalMesh(4)

cell = mesh.ufl_cell()

elem = fd.FiniteElement("CG", cell, 1)    # primal element
elem2 = fd.MixedElement([elem, elem])     # mixed element
elem22 = fd.MixedElement([elem2, elem2])  # nested mixed element

fs22 = fd.FunctionSpace(mesh, elem22)

print(f"elem: {elem}")
print(f"elem2: {elem2}")
print(f"elem22: {elem22}")
print(f"fs22.ufl_element(): {fs22.ufl_element()}")
print("\n")
print(f"elem.sub_element(): {elem.sub_elements()}")
print(f"elem2.sub_element(): {elem2.sub_elements()}")
print(f"elem22.sub_element(): {elem22.sub_elements()}")
print(f"fs22.ufl_element().sub_element(): {fs22.ufl_element().sub_elements()}")
print("\n")
print(f"elem.num_sub_element(): {elem.num_sub_elements()}")
print(f"elem2.num_sub_element(): {elem2.num_sub_elements()}")
print(f"elem22.num_sub_element(): {elem22.num_sub_elements()}")
print(f"fs22.ufl_element().num_sub_element(): {fs22.ufl_element().num_sub_elements()}")

assert fs22.ufl_element() != elem22 # passes
