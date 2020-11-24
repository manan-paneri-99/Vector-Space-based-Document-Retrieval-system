from query_processer import search


def test_queries(open_web, use_zones, enable_query_relaxation=1):
    """
    Prints the top 10 doc_ids with their score and titles
    :param open_web: set to True if results are to be opened on browser window
    :param use_zones: set to True to enable Zonal indexing
    :param enable_query_relaxation:
    :return:
    """
    # Taking query input
    query = input("Type the query: ")
    search(query, open_web=open_web, use_zones=use_zones, enable_query_relaxation=enable_query_relaxation)

if __name__ == "__main__":
    test_queries(open_web=False, use_zones=False, enable_query_relaxation=1)
