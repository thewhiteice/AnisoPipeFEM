# %%
import yaml

cfg = yaml.safe_load(
    """
    geometry:
        inner_radius: 328.0e-3
    name:
        name1: defaultIII
        name2: 康凯2022
    """)

print(cfg["geometry"]["inner_radius"])  # 0.328
print(type(cfg["geometry"]["inner_radius"]))  # <class 'float'>
print(cfg["name"])
print(cfg["name"]["name1"])
print(cfg["name"]["name2"])

# %%
print(type(yaml.safe_load("x: 1950.0e6")["x"]))   # str
print(type(yaml.safe_load("x: 1950.0e+6")["x"]))  # float
