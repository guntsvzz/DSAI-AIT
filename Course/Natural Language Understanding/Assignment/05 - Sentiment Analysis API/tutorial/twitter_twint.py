import twint
# pip3 install twint
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

def scraper(keyword):
    c = twint.Config()
    c.Search = keyword
    c.Limit = 5
    c.Store_object = True

    twint.run.Search(c)
    output = twint.output.tweets_list

    return output

if __name__ =="__main__":
    tweets = scraper("Bitcoin")
    print('--'*30)
    for tweet in tweets:
        print(tweet.tweet)