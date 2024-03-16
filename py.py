import turtle
score_board = Turtle()
    score_board.hideturtle()
    score_board.setposition(0, 250)
    score_board.color('white')
    score_board.penup()

    score = 0
    a = [0,1,2,3,4,5,6]
    for _ in a:
        while  _ < 15:
            score += 1
            print(score)

    score_board.write(arg=f"SCORE = {score}", move=False, align='center', font=('Arial', 30, 'normal'))