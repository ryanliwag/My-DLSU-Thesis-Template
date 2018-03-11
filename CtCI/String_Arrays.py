def Is_Unique(string):
    """Implement an algorithm to determine if a string has all unique characters. What if you
    cannot use additional data structures"""
    string_arr = list(string)
    
    for i in range(len(string_arr)):
        tmp = string_arr.pop()
        if tmp in string_arr:
            return False
    
    return True


def Check_Permutation(str1, str2):
    """Given two strings, write a method to decide if one is a permutation of the
    other."""

    str1_arr, str2_arr = list(str1), list(str2)
    
    if len(str1_arr) == len(str2_arr):
        for i in range(len(str1_arr)):
            if str1_arr[i] in str2_arr:
                str2_arr.remove(str1_arr[i])
            else:
                return False
        return True
    else:
        return False

def URLify(string):
    """Write a method to replace all spaces in a string with '%20: You may assume that the string
       has sufficient space at the end to hold the additional characters, and that you are given the "true"
       length of the string."""
    tmp_string = []
    for i in range(len(string) - 1):
        if string[i] == " ":
            if string[i + 1] != " ":
                tmp_string.append("%")        
        else:
            tmp_string.append(string[i])
    tmp_string = "".join(tmp_string)
    
    return tmp_string
    

def Palindrome_Permutation(str1):
    """Given a string, write a function to check if it is a permutation of a palin-
       drome. A palindrome is a word or phrase that is the same forwards and backwards. A permutation
       is a rea rrangement of letters. The palindrome does not need to be limited to just dictionary words."""
    str1=str1.replace(" ","")
    item_list = list(str1.lower())
    uniq_items = set(item_list)
    
    counter = 0
    for i in uniq_items:
        if item_list.count(i) % 2 != 0:
            counter += 1
    
    if counter > 1:
        return False
    else:
        return True

def is_one_away(first: str, other: str) -> bool:
    """Given two strings, check if they are one edit away. An edit can be any one of the following.
    1) Inserting a character
    2) Removing a character
    3) Replacing a character"""

    skip_difference = {
        -1: lambda i: (i, i+1),  # Delete
        1: lambda i: (i+1, i),  # Add
        0: lambda i: (i+1, i+1),  # Modify
    }
    try:
        skip = skip_difference[len(first) - len(other)]
    except KeyError:
        return False  

    for i, (l1, l2) in enumerate(zip(first, other)):
        print(i)
        if l1 != l2:
            i -= 1  
            break

    remain_first, remain_other = skip(i + 1)
    return first[remain_first:] == other[remain_other:]
