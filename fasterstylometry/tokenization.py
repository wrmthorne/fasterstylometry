import polars as pl


PRONOUNS = {
    "he", "her", "hers", "herself", "him", "himself", "his", "i", "me", "mine", "my", "myself", 
    "our", "ours", "ourselves", "she", "thee", "their", "them", "themselves", "they", "thou", 
    "thy", "thyself", "us", "we", "ye", "you", "your", "yours", "yourself"
}


# Default tokeniser 
tokenise_remove_pronouns_en = (
    pl.col('text')
    .str.to_lowercase()
    .str.replace_all("[`']", '')
    .str.extract_all('\w+')
    .list.eval(
        pl.element().filter(
            (~pl.element().is_in(PRONOUNS)) & (~pl.element().str.contains('\d'))
        ),
        parallel=True
    )
)