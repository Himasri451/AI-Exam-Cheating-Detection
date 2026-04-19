def log_event(message):
    with open("log.txt", "a") as file:
        file.write(message + "\n")

    print("LOG:", message)
