

def main():
    print("""
Societal Collapse.
Choose an option:

Societal collapse by...
[1] Anarchy*
[2] Deadly pathogen*
[3] Nuclear war*
[4] Biological war*
[5] Astroid*
[6] San Andreas fault*
[7] Super volcano*
[8] False void
[9] Gamma ray burst*

*not implemented""")
    eow_option = input("Choose an option:  ")
    population = input("What is the population of your society? [1k, 10k, 100k, 1m] ")
    political_stability = input("How is the political stability? [low, med, high] ")
    police_strength = input("What is the strength/effectiveness of the police? [low, med, high] ")
    weapon_access = input("Do citizens have easy access to weapons? [yes, no] ")
    ideology = input("What is the ideology? [democracy, anarchism, facism]")

    """
    use switch statement
    """

    """
    - Population: 1k, 10k, 100k, 1m (4 values)
    - Political stability: Low, Medium, High (3 values)
    - Police strength: Low, Medium, High (3 values)
    - Easy weapon access: Yes, No (2 values)
    - Ideology: Democracy, Anarchism, Fascism (3 values)

    
    There are 4 values for the population variable,
    3 values for political stability and police
    strength, 2 values for easy weapon access, and
    3 values for ideology, there would be a total
    of 4 x 3 x 3 x 2 x 3 = 216 possible combinations.

    Assign each combination a unique ID and store the
    inputs and outputs associated with each ID in a
    table or spreadsheet.
    """

if __name__ == "__main__":
    main()
