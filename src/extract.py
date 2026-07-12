import lief 
def extract_features(pe_path):
    pe = lief.parse(pe_path)
    if pe is None:
        return None
    features = {}
    features["path"] = pe_path
    features["num_sections"] = len(pe.sections)
    section_names = []
    section_entropies = {}
    for section in pe.sections:
        section_names.append(section.name)
        section_entropies[section.name] = round(section.entropy, 2)
    features["section_names"] = section_names
    features["section_entropies"] = section_entropies 
    imports = {}
    if pe.has_imports:
        for lib in pe.imports:
            func_names = []
            for entry in lib.entries:
                if entry.name:
                    func_names.append(entry.name)
            imports[lib.name] = func_names
    features["imports"] = imports
    return features
if __name__ == "__main__":
    result = extract_features("data/test_pe/putty.exe")
    print("num_sections:", result["num_sections"])
    print("sections:", result["section_names"])
    print("num DLLs imported:", len(result["imports"]))
    print("section_entropies:", result["section_entropies"])
 