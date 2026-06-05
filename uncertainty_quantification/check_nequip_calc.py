from nequip.integrations.ase import NequIPCalculator
import inspect
meths = [m for m in dir(NequIPCalculator) if not m.startswith("_")]
print("methods:", meths)
# check from_compiled_model
print("\nfrom_compiled_model sig:")
print(inspect.signature(NequIPCalculator.from_compiled_model))
print("\nfrom_compiled_model src:")
print(inspect.getsource(NequIPCalculator.from_compiled_model)[:2000])
