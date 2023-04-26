import ml.anarchy.anarchy as anarchy

def main():
    print("""
Societal Collapse Simulator.
Choose an option:

Societal collapse by...
[1] Anarchy
[2] Gamma ray burst*

*not implemented""")
    eow_option = input("Choose an option:  ")
    
    if eow_option == "1":
        anarchy.ask()
    else:
        print("Invalid option!!")
        return

if __name__ == "__main__":
    main()
