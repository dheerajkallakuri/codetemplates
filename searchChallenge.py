def SearchingChallenge(str):
    # __define-ocg__: Initialize the variable to store the longest pattern
    varOcg = ""

    max_pattern = ""
    n = len(str)
    
    # Function to check if a substring repeats
    def is_repeating(sub, s):
        # Count occurrences of the substring in the main string
        return s.count(sub) > 1

    # Iterate over all possible substrings of the string
    for length in range(1, n//2 + 1):  # Length of the pattern
        for i in range(n - 2 * length + 1):
            substring = str[i:i+length]
            if is_repeating(substring, str):
                # If the found pattern is longer, update the max_pattern
                if len(substring) > len(max_pattern):
                    max_pattern = substring
    
    # If a pattern was found, return 'yes' followed by the pattern
    if max_pattern:
        varOcg = f"yes {max_pattern}"
    else:
        varOcg = "no null"
    
    return varOcg

# Test cases
print(SearchingChallenge("aabecaa"))
