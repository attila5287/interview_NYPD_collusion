from __future__ import print_function, division

class Card:
    def __init__(
        self,
        suit = 0,
        rank = 2 
    ):
    self.suit = suit
    self.rank = rank
    



    suit_names = ['Clubs','Diamonds','Hearts','Spades']
    rank_names = [ None,'Ace','2','3','4','5','6','7','8','9','10','Jack','Queen','King']
    
    def __str__(self):
        return '%s of  %s %s' % (Card.rank_names[self.rank],
        Card.suit_names[self.suit]        
        )





if __name__ == '__main__':

    # deck = Deck()
    # deck.shuffle()
    # hand = Hand()
    # print(find_defining_class(hand, 'shuffle'))
    # deck.move_cards(hand, 5)
    # hand.sort()
    # print(hand)