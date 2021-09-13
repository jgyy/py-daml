"""
Python Crash Course Exercises
"""
# ** What is 7 to the power of 4?**
print(7 ** 4)

# ** Split this string:**
S = "Hi there Sam!"
print(S.split())

# ** Use .format() to print the following string: **
PLANET = "Earth"
DIAMETER = 12742
print(f"The diameter of {PLANET} is {DIAMETER} kilometers.")

# ** Given this nested list, use indexing to grab the word "hello" **
lst = [1, 2, [3, 4], [5, [100, 200, ["hello"]], 23, 11], 1, 7]
print(lst[3][1][2][0])

# ** Given this nested dictionary grab the word "hello" **
d = {
    "k1": [
        1,
        2,
        3,
        {
            "tricky": [
                "oh",
                "man",
                "inception",
                {"target": [1, 2, 3, "hello", "world", "python"]},
            ]
        },
    ]
}
print(d["k1"][3]["tricky"][3]["target"][3])

# ** Create a function that grabs the email website domain from a string in the form: **
domain_get = lambda x: x.split("@")[1]
print(domain_get("user@domain.com"))

# Create a basic function that returns True if the word 'dog' is contained in the input string.
find_dog = lambda x: "dog" in x.lower()
print(find_dog("Is there a dog here?"))

# Create a function that counts the number of times the word "dog" occurs in a string.
count_dog = lambda x: x.split().count("dog")
print(count_dog("This dog runs faster than the other dog dude!"))

# Use lambda expressions and the filter() function to filter out words from a list.
seq = ["soup", "dog", "salad", "cat", "great"]
print(list(filter(lambda x: x[0].lower() == "s", seq)))

# Final Problem
def caught_speeding(speed, is_birthday):
    """
    You are driving a little too fast, and a police officer stops you. Write a function
    to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket".
    If your speed is 60 or less, the result is "No Ticket". If speed is between 61
    and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is
    our birthday -- on your birthday, "Big Ticket".
    Unless it is yyour speed can be 5 higher in all cases.
    """
    if is_birthday:
        speed -= 5
    if speed <= 60:
        ticket = "No Ticket"
    elif 60 < speed <= 80:
        ticket = "Small Ticket"
    elif speed > 80:
        ticket = "Big Ticket"
    return ticket


print(caught_speeding(81, True))
print(caught_speeding(81, False))
