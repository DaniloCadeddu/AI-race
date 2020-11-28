import pygame 
import neat
import os 
import random
from pygame import mixer

pygame.mixer.init()
pygame.font.init()

win_width = 500
win_height = 600
gen = 0
car_img = (pygame.image.load(os.path.join("imgs", "racing-car.png")))
wheels_img = (pygame.image.load(os.path.join("imgs", "wheels.png")))
base_img = (pygame.image.load(os.path.join("imgs", "bg.png")))
stat_font = pygame.font.SysFont("comicsans", 50)

pygame.display.set_caption("car race")
icon = pygame.image.load("./imgs/racing-car.png")
pygame.display.set_icon(icon)
""" mixer.music.load('')
mixer.music.play(-1) """

    
# Creating the Car class where we can define position, velocity and the physics of its motion
class Car :
    img = car_img

    def __init__(self, x, y) :
        self.x = x
        self.y = y
        #self.tilt = 0
        self.tick_count = 0 # time
        self.vel = 0
        self.height = y
        self.img = car_img

    def turn(self) :
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self) :
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5*self.tick_count**2 # motion under constant acceleration

        if d >= 16 :
            d = 16
        if d < 0 :
            d -= 2

        self.y = self.y + d

    
    def draw(self, win) :
        win.blit(self.img, (self.x, self.y))
    
    # The mask is what we need to detect collision
    def get_mask(self) :
        return pygame.mask.from_surface(self.img)

# This class creates the wheel that in this case are the obstacles of this game
class Wheels :
    gap = 170
    vel = 5

    def __init__(self, x) :
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.wheel_top = pygame.transform.flip(wheels_img, False, True)
        self.wheel_bottom = wheels_img

        self.passed = False
        self.set_height()

    def set_height(self) :
        self.height = random.randrange(50, 450)
        self.top = self.height - self.wheel_top.get_height()
        self.bottom = self.height + self.gap
    
    def move(self) :
        self.x -= self.vel
    
    def draw(self, win) :
        win.blit(self.wheel_top, (self.x, self.top))
        win.blit(self.wheel_bottom, (self.x, self.bottom))
    
    # Detecs collision with the car
    def collide(self, car) :
        car_mask = car.get_mask()
        top_mask = pygame.mask.from_surface(self.wheel_top)
        bottom_mask = pygame.mask.from_surface(self.wheel_bottom)

        top_offset = (self.x - car.x, self.top - round(car.y)) # The offset between the car and the wheel
        bottom_offset = (self.x - car.x, self.bottom - round(car.y))

        b_point = car_mask.overlap(bottom_mask, bottom_offset)
        t_point = car_mask.overlap(top_mask, top_offset)

        if t_point or b_point :
            return True
        return False
# Create the moving road background
class Base :
    vel = 5
    width = base_img.get_width()
    img = base_img

    def __init__ (self, y) :
        self.y = y
        self.x1 = 0
        self.x2 = self.width
    
    def move(self) :
        self.x1 -= self.vel
        self.x2 -= self.vel

        if self.x1 + self.width < 0 :
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0 : 
            self.x2 = self.x1 + self.width
        
    def draw(self, win) :
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))

# Draw everything, window, wheels, car, background
def draw_window(win, car, wheels, backg, score, gen) :
    
    backg.draw(win)

    for wheel in wheels :
        wheel.draw(win)
    
    text = stat_font.render("Score: " + str(score), 1, (0,0,0))
    win.blit(text, (win_width - 10 - text.get_width(), 10))
    
    text = stat_font.render("Gen: " + str(gen), 1, (0,0,0))
    win.blit(text, (10, 10))
    
    
    for car in car :
        car.draw(win)
    pygame.display.update()

def main(genomes, config) :
    win = pygame.display.set_mode((win_width, win_height))
    global gen
    gen += 1
    nets = []
    ge = []
    cars = []
    
    # Config the cars with their genomes and create their neural net
    for _,g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(230, 350))
        g.fitness = 0
        ge.append(g)

    backg = Base(0)
    wheels = [Wheels(600)]
      
    clock = pygame.time.Clock()
    score = 0
    run = True
    fps = 30
    while run :
        clock.tick(fps)

        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                run = False
                pygame.quit()
                quit()

        wheel_ind = 0
        if len(cars) > 0 :
            if len(wheels) > 1 and cars[0].x > wheels[0].x + wheels[0].wheel_top.get_width() :
                wheel_ind = 1
        else :
            run = False
            break
        
        # A car gain points if stays alive
        for x, car in enumerate(cars) :
            car.move()
            ge[x].fitness += 0.1

            # Depends on the value output a car decide if it has to turn or not
            output = nets[x].activate((car.y, abs(car.y - wheels[wheel_ind].height), abs(car.y - wheels[wheel_ind].bottom)))
            if output[0] > 0.5 :
                car.turn()

        add_wheels = False
        rem = []
        for wheel in wheels :
            for x, car in enumerate(cars) : # A car that collide loses points
                if wheel.collide(car) :
                    ge[x].fitness -= 1
                    cars.pop(x)
                    nets.pop(x)
                    ge.pop(x)
          
                if not wheel.passed and wheel.x < car.x :
                    wheel.passed = True
                    add_wheels = True
            
            if wheel.x + wheel.wheel_top.get_width() < 0 :
                rem.append(wheel)


            wheel.move()
        # Cars that pass an obstacle get more points
        if add_wheels :
            score += 1
            for g in ge :
                g.fitness += 5
            wheels.append(Wheels(600))
            
        for r in rem :
            wheels.remove(r)
        
        # Delete cars that go out of the screen
        for x, car in enumerate(cars) :
            if car.y + car.img.get_height() >= 730 or car.y < 0 :
                cars.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        backg.move()     
        draw_window(win, cars, wheels, backg, score, gen)


# All the algorithm configurations
def run(config_path) :
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)


if __name__ == "__main__" :
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedfoward.txt")
    run(config_path)