import lief  # Import the LIEF library for parsing PE (Portable Executable) files
def extract_features(pe_path):  # Define a function that takes a path to a PE file
    pe = lief.parse(pe_path)  # Parse the file at pe_path into a LIEF PE object
    if pe is None:  # Check whether parsing failed (invalid or unreadable file)
        return None  # Return None if the file could not be parsed
    features = {}  # Create an empty dictionary to store extracted features
    features["path"] = pe_path  # Store the input file path in the features dict
    features["num_sections"] = len(pe.sections)  # Record how many sections the PE has
    section_names = []  # Initialize a list to collect section names
    section_entropies = {}  # Initialize a dict mapping section names to entropy values
    for section in pe.sections:  # Iterate over each section in the PE
        section_names.append(section.name)  # Add this section's name to the list
        section_entropies[section.name] = round(section.entropy, 2)  # Store rounded entropy keyed by section name
    features["section_names"] = section_names  # Save the list of section names to features
    features["section_entropies"] = section_entropies  # Save the section entropy map to features
    imports = {}  # Initialize a dict to store imported DLLs and their functions
    if pe.has_imports:  # Only process imports if the PE declares an import table
        for lib in pe.imports:  # Iterate over each imported library (DLL)
            func_names = []  # Initialize a list for function names from this DLL
            for entry in lib.entries:  # Iterate over each import entry in the library
                if entry.name:  # Skip entries without a resolved name (e.g. ordinal-only)
                    func_names.append(entry.name)  # Add the imported function name to the list
            imports[lib.name] = func_names  # Map this DLL name to its list of imported functions
    features["imports"] = imports  # Save the imports dictionary to features
    return features  # Return the completed features dictionary
result = extract_features("data/test_pe/putty.exe")  # Call extract_features on a sample PE file
print("num_sections:", result["num_sections"])  # Print the number of sections from the result
print("sections:", result["section_names"])  # Print the list of section names
print("num DLLs imported:", len(result["imports"]))  # Print how many DLLs are in the import table
