import itertools
import time

import pandas as pd
from tqdm import tqdm  # type: ignore

WORD_LENGTH = 5
NUM_WORDS = 5


def _bitval(word: str) -> int:
    """Turn word of chars into 26 bit integer where each char corresponds to least significant bit place in int value.
    Can do this since only 1 of each letter max can be present so only 0/1 bit needed per letter.
    e.g. abled a-place=0,b-place=1, l-place=12,e-place=5, d-place=4, so bitval=b'00000000000000100000011011"""
    bitval: int = 0
    for char in word:
        # get place in alphabet
        alph_place = ord(char) - ord("a")
        # add bit at alphabet place
        bitval |= 1 << alph_place
    # print(f"{bitval=:026b}")
    return bitval


def remove_duplicates(words: list[str]) -> list[str]:
    """Clear out words that have 2 or more of one letter since can't be part of 5 letter per word five clique."""
    return [word for word in words if len(set(word)) == WORD_LENGTH]


def get_anagrams(words: list[str]) -> dict[int, set[str]]:
    """Get anagrams or words that share all the same letters by computing bitval which is same for anagrams then make mapping of bitvals to associated words."""
    anagrams: dict[int, set[str]] = {}
    for word in words:
        word_val = _bitval(word)
        if word_val not in anagrams:
            anagrams[word_val] = set()
        anagrams[word_val].add(word)
    return anagrams


def get_graph(word_vals: list[int]) -> dict[int, set[int]]:
    """Construct graph by getting each word value and comparing it to words after it (for one way/upper traingular relationship to avoid duplicate comparisons).
    If wordvals bitwise AND is 0 or if words have no common letters, make them neighbors in graph."""
    graph: dict[int, set[int]] = {word_val: set() for word_val in word_vals}
    for idx, base_set in tqdm(
        enumerate(word_vals), total=len(word_vals), desc="Constructing Graph"
    ):
        for compare_set in word_vals[idx + 1 :]:
            # Bitwise AND of unique character bitvals, if 2 words share a letter, aka 2 bitvals both have a 1 in same place, this will be nonzero
            if base_set & compare_set == 0:
                graph[base_set].add(compare_set)
    return graph


def find_cliques(graph: dict[int, set[int]]) -> list[list[int]]:
    """Using neighbors of graph, find n-clique where n points can all access each other directly as neighbors.
    For this problem this means all n words are neighbors to one another so none share any common letters.
    Pruning done by"""
    prune: set[int] = set()

    cliques: list[list[int]] = []
    for (i_wordval, i_neighbors) in tqdm(
        sorted(graph.items()), total=len(graph), desc="Find cliques"
    ):
        for j_wordval in i_neighbors:
            j_neighbors = graph[j_wordval]
            # check if i and j intersection path already checked/pruned ou to avoid needless recalculating
            if (i_wordval | j_wordval) in prune:
                continue

            # Get neighbors of i and j by getting intersection of sets of neighbors
            # e.g. i_n ={1,2,3}, j_n={2,3,4} i_n&j_n={2,3}
            ij_neighbors = i_neighbors & j_neighbors
            # If no other neighbors of words i,j besides themselves no way they can form clique so prune combo out for future searches, stop search here
            if len(ij_neighbors) < 3:
                prune.add(i_wordval | j_wordval)
                continue

            have_ij = False
            # Search ij intersect neighbor k, get k's neighbors by looking at graph, then get intesection of k neighbors with ij neighbors
            for k_wordval in ij_neighbors:
                k_neighbors = graph[k_wordval]
                if (i_wordval | j_wordval | k_wordval) in prune:
                    continue
                ijk_neighbors = ij_neighbors & k_neighbors
                if len(ijk_neighbors) < 2:
                    prune.add(i_wordval | j_wordval | k_wordval)
                    continue

                have_ijk = False
                for l_wordval in ijk_neighbors:
                    ijkl_neighbors = ijk_neighbors & graph[l_wordval]

                    # all remaining neighbors form a 5-clique with i, j, k, and l
                    for r_wordval in ijkl_neighbors:
                        cliques.append(
                            [i_wordval, j_wordval, k_wordval, l_wordval, r_wordval]
                        )
                        have_ij = True
                        have_ijk = True

                # Now update pruning if deeper search still no cliques
                if not have_ijk:
                    # we didn't find anything on this branch, prune it
                    prune.add(i_wordval | j_wordval | k_wordval)
            if not have_ij:
                # we didn't find anything on this branch, prune it
                prune.add(i_wordval | j_wordval)

    return cliques


def get_words(
    cliques: list[list[int]], anagrams: dict[int, list[str]]
) -> list[list[str]]:
    """From wordvals found in cliques, get back possible words with all anagrams using anagram mapping."""
    word_lists: list[list[str]] = []
    for clique in cliques:
        all_possible_words = [anagrams[wordval] for wordval in clique]
        # Get all possible expanded anagrams per set of n words
        # e.g. all_possible_words = [{hello},{carts,drugs},{hikes, bikes}, {seeya}]->
        # [hello,carts,hikes,seeya],[hello,carts,bikes,seeya], [hello, drugs, bikes, seeya], [hello,drugs, hikes, seeya]
        all_cliques = list(itertools.product(*all_possible_words))
        for wclique in all_cliques:
            word_lists.append([word for word in wclique])  # type: ignore
    print(word_lists)
    return word_lists


def main():
    valid_word_list: list[str] = (
        pd.read_csv("data/wordle_answers.csv").to_numpy().flatten().tolist()
        + pd.read_csv("data/wordle_list.csv").to_numpy().flatten().tolist()
    )
    print(
        f"Initial {len(valid_word_list)} first {valid_word_list[0]} last {valid_word_list[-1]}"
    )
    word_list = remove_duplicates(valid_word_list)
    print(f"Initial rm {len(word_list)} first {word_list[0]} last {word_list[-1]}")
    anagram_map = get_anagrams(word_list)
    print(f"{len(anagram_map)=}")
    wordvals = sorted(list(anagram_map.keys()))
    print(f"{len(wordvals)=} first {wordvals[0]} last {wordvals[-1]}")
    graph = get_graph(wordvals)
    print(f"{len(graph)=}")
    cliques = find_cliques(graph)
    print(f"Cliques {len(cliques)}")
    get_words(cliques, anagram_map)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Total time: {time.perf_counter()-start} seconds")
