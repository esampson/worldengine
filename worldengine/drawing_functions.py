"""
This file should contain only functions that operates on pixels, not on images,
so no references to PIL are necessary and the module can be used also through
Jython
"""

import numpy
import sys
import time
from PIL import Image
from worldengine.common import get_verbose


# -------------------
# Reusable functions
# -------------------


def gradient(value, low, high, low_color, high_color):
    lr, lg, lb = low_color
    if high == low:
        return lr, lg, lb, 255
    _range = float(high - low)
    _x = float(value - low) / _range
    _ix = 1.0 - _x
    hr, hg, hb = high_color
    r = int(lr * _ix + hr * _x)
    g = int(lg * _ix + hg * _x)
    b = int(lb * _ix + hb * _x)
    return r, g, b, 255


def rgba_to_rgb(rgba):
    r, g, b, a = rgba
    return r, g, b


def draw_rivers_on_image(world, target, factor=1):
    """Draw only the rivers, it expect the background to be in place
    """

    for y in range(world.height):
        for x in range(world.width):
            if world.is_land((x, y)) and (world.river_map[x, y] > 0.0):
                for dx in range(factor):
                    for dy in range(factor):
                        target.set_pixel(x * factor + dx, y * factor + dy, (0, 0, 128, 255))
            if world.is_land((x, y)) and (world.lake_map[x, y] != 0):
                for dx in range(factor):
                    for dy in range(factor):
                        target.set_pixel(x * factor + dx, y * factor + dy, (0, 100, 128, 255))


# -------------------
# Drawing ancient map
# -------------------

def _find_land_borders(world, factor):
    _ocean = numpy.zeros((factor * world.height, factor * world.width), dtype=bool)
    _borders = numpy.zeros((factor * world.height, factor * world.width), dtype=bool)

    #scale ocean
    for y in range(world.height * factor):  # TODO: numpy
        for x in range(world.width * factor):
            if world.ocean[int(y / factor), int(x / factor)]:
                _ocean[y, x] = True

    def my_is_ocean(pos):
        x, y = pos
        return _ocean[y, x]

    for y in range(world.height * factor):
        for x in range(world.width * factor):
            if not _ocean[y, x] and world.tiles_around_factor(factor, (x, y), radius=1, predicate=my_is_ocean):
                _borders[y, x] = True
    return _borders


def _find_outer_borders(world, factor, inner_borders):
    _ocean = numpy.zeros((factor * world.height, factor * world.width), dtype=bool)
    _borders = numpy.zeros((factor * world.height, factor * world.width), dtype=bool)

    #scale ocean
    for y in range(world.height * factor):  # TODO: numpy
        for x in range(world.width * factor):
            if world.ocean[int(y / factor)][int(x / factor)]:
                _ocean[y, x] = True

    def is_inner_border(pos):
        x, y = pos
        return inner_borders[y, x]

    for y in range(world.height * factor):
        for x in range(world.width * factor):
            if _ocean[y, x] and not inner_borders[y, x] and world.tiles_around_factor(factor, (x, y), radius=1, predicate=is_inner_border):
                _borders[y, x] = True
    return _borders


def _find_mountains_mask(world, factor):
    _mask = numpy.zeros((factor * world.height, factor * world.width), dtype=float)
    for y in range(factor * world.height):
        for x in range(factor * world.width):
            if world.is_mountain((int(x / factor), int(y / factor))):
                v = len(world.tiles_around((int(x / factor), int(y / factor)),
                                           radius=3,
                                           predicate=world.is_mountain))
                if v > 32:
                    _mask[y, x] = v / 4.0  # force conversion to float, Python 2 will *not* do it automatically
    return _mask


def _mask(world, predicate, factor):
    _mask = numpy.zeros((factor * world.height, factor * world.width), dtype=float)
    for y in range(factor * world.height):
        for x in range(factor * world.width):
            xf = int(x / factor)
            yf = int(y / factor)
            if predicate((xf, yf)):
                v = len(
                    world.tiles_around((xf, yf), radius=1,
                                       predicate=predicate))
                if v > 5:
                    _mask[y, x] = v
    return _mask

def _createmask(world, predicate, factor):
    width = world.width * factor
    height = world.height * factor
    _mask = Image.new("RGBA", (width,height))
    for y in range(height):
        for x in range(width):
            xf = int(x / factor)
            yf = int(y / factor)
            if predicate((xf, yf)):
                _mask.putpixel((x,y),(0,0,0,255))
    return _mask  

def _draw_cool_desert(pixels, x, y, w, h):
    c = (72, 72, 53, 255)
    # c2 = (219, 220, 200, 255)  # TODO: not used?

    pixels[x - 1, y - 2] = c
    pixels[x - 0, y - 2] = c
    pixels[x + 1, y - 2] = c
    pixels[x + 1, y - 2] = c
    pixels[x + 2, y - 2] = c
    pixels[x - 2, y - 1] = c
    pixels[x - 1, y - 1] = c
    pixels[x - 0, y - 1] = c
    pixels[x + 4, y - 1] = c
    pixels[x - 4, y - 0] = c
    pixels[x - 3, y - 0] = c
    pixels[x - 2, y - 0] = c
    pixels[x - 1, y - 0] = c
    pixels[x + 1, y - 0] = c
    pixels[x + 2, y - 0] = c
    pixels[x + 6, y - 0] = c
    pixels[x - 5, y + 1] = c
    pixels[x - 0, y + 1] = c
    pixels[x + 7, y + 1] = c
    pixels[x + 8, y + 1] = c
    pixels[x - 8, y + 2] = c
    pixels[x - 7, y + 2] = c

def _stamp(image, stamp, x, y, w, h):
    stamp = stamp.resize([int(w), int(h)],Image.ANTIALIAS)
    left = x - int(w/2)
    top = y - int(h)
    image._paste(stamp, (left, top))

def _texture(image, texture, mask):
    size = image._size()
    width = size[0]
    height = size[1]
    width = int(width)
    height = int(height)
    tx = Image.new("RGBA", (width,height))
    xcount = int(width / texture.size[0])
    if xcount < float(width) / texture.size[0]:
        xcount = xcount + 1
    ycount = int(height / texture.size[1])
    if ycount != float(height) / texture.size[1]:
        ycount = ycount + 1
    for x in range(0,xcount):
        for y in range(0,ycount):
            left = x * texture.size[0]
            top = y * texture.size[1]
            tx.paste(texture, (left, top))
    image._fill(tx, mask)

# TODO: complete and enable this one
def _dynamic_draw_a_mountain(pixels, rng, x, y, w=3, h=3):
    # mcl = (0, 0, 0, 255)  # TODO: No longer used?
    # mcll = (128, 128, 128, 255)
    mcr = (75, 75, 75, 255)
    # left edge
    last_leftborder = None
    for mody in range(-h, h + 1):
        bottomness = (float(mody + h) / 2.0) / w

        min_leftborder = int(bottomness * w * 0.66)
        if not last_leftborder == None:
            min_leftborder = max(min_leftborder, last_leftborder - 1)
        max_leftborder = int(bottomness * w * 1.33)
        if not last_leftborder == None:
            max_leftborder = min(max_leftborder, last_leftborder + 1)
        leftborder = int(bottomness * w) + rng.randint(-2, 2)/2
        if leftborder < min_leftborder:
            leftborder = min_leftborder
        if leftborder > max_leftborder:
            leftborder = max_leftborder
        last_leftborder = leftborder

        darkarea = int(bottomness * w / 2)
        lightarea = int(bottomness * w / 2)
        for itx in range(darkarea, leftborder + 1):
            pixels[x - itx, y + mody] = gradient(itx, darkarea, leftborder,
                                                 (0, 0, 0), (64, 64, 64))
        for itx in range(-darkarea, lightarea + 1):
            pixels[x + itx, y + mody] = gradient(itx, -darkarea, lightarea,
                                                 (64, 64, 64), (128, 128, 128))
        for itx in range(lightarea, leftborder):
            pixels[x + itx, y + mody] = (181, 166, 127, 255)  # land_color
    # right edge
    last_modx = None
    for mody in range(-h, h + 1):
        bottomness = (float(mody + h) / 2.0) / w
        min_modx = int(bottomness * w * 0.66)
        if not last_modx == None:
            min_modx = max(min_modx, last_modx - 1)
        max_modx = int(bottomness * w * 1.33)
        if not last_modx == None:
            max_modx = min(max_modx, last_modx + 1)
        modx = int(bottomness * w) + numpy.random.randint(-2, 2)/2
        if modx < min_modx:
            modx = min_modx
        if modx > max_modx:
            modx = max_modx
        last_modx = modx
        pixels[x + modx, y + mody] = mcr


def _draw_a_mountain(pixels, x, y, w=3, h=3):
    # mcl = (0, 0, 0, 255)  # TODO: No longer used?
    # mcll = (128, 128, 128, 255)
    mcr = (75, 75, 75, 255)
    # left edge
    for mody in range(-h, h + 1):
        bottomness = (float(mody + h) / 2.0) / w
        leftborder = int(bottomness * w)
        darkarea = int(bottomness * w / 2)
        lightarea = int(bottomness * w / 2)
        for itx in range(darkarea, leftborder + 1):
            pixels[x - itx, y + mody] = gradient(itx, darkarea, leftborder,
                                                 (0, 0, 0), (64, 64, 64))
        for itx in range(-darkarea, lightarea + 1):
            pixels[x + itx, y + mody] = gradient(itx, -darkarea, lightarea,
                                                 (64, 64, 64), (128, 128, 128))
        for itx in range(lightarea, leftborder):
            pixels[x + itx, y + mody] = (181, 166, 127, 255)  # land_color
    # right edge
    for mody in range(-h, h + 1):
        bottomness = (float(mody + h) / 2.0) / w
        modx = int(bottomness * w)
        pixels[x + modx, y + mody] = mcr        


def draw_ancientmap(world, target, resize_factor=1,
                    sea_color=(212, 198, 169, 255),
                    draw_biome = True, draw_rivers = True, draw_mountains = True,
                    draw_outer_land_border = False, verbose = False):
    rng = numpy.random.RandomState(world.seed)  # create our own random generator

    if verbose:
        start_time = time.time()
    print("Loading bitmaps")

    Mountains = [Image.open("worldengine/data/mountain1.png"),
                 Image.open("worldengine/data/mountain2.png"),
                 Image.open("worldengine/data/mountain3.png"),
                 Image.open("worldengine/data/mountain4.png"),
                 Image.open("worldengine/data/mountain5.png")]

    Hills = [Image.open("worldengine/data/hill1.png"),
             Image.open("worldengine/data/hill2.png"),
             Image.open("worldengine/data/hill3.png")]
    
    Deserts = [Image.open("worldengine/data/desert1.png"),
               Image.open("worldengine/data/desert2.png"),
               Image.open("worldengine/data/desert3.png")]

    Decids = [Image.open("worldengine/data/decid1.png"),
              Image.open("worldengine/data/decid2.png"),
              Image.open("worldengine/data/decid3.png"),
              Image.open("worldengine/data/decid4.png"),
              Image.open("worldengine/data/decid5.png"),
              Image.open("worldengine/data/decid6.png")]

    Pines = [Image.open("worldengine/data/pine1.png"),
             Image.open("worldengine/data/pine2.png"),
             Image.open("worldengine/data/pine3.png"),
             Image.open("worldengine/data/pine4.png"),
             Image.open("worldengine/data/pine5.png"),
             Image.open("worldengine/data/pine6.png")]

    Jungles = [Image.open("worldengine/data/palm1.png"),
               Image.open("worldengine/data/palm2.png"),
               Image.open("worldengine/data/palm3.png"),
               Image.open("worldengine/data/palm4.png"),
               Image.open("worldengine/data/palm5.png"),
               Image.open("worldengine/data/palm6.png")]

    DryTropicals = [Image.open("worldengine/data/DT1.png"),
                    Image.open("worldengine/data/DT2.png"),
                    Image.open("worldengine/data/DT3.png"),
                    Image.open("worldengine/data/DT4.png"),
                    Image.open("worldengine/data/DT5.png"),
                    Image.open("worldengine/data/DT6.png")]

    Ice = Image.open("worldengine/data/ice.png")

    Tundra = Image.open("worldengine/data/tundra.png")

    Parklands = Image.open("worldengine/data/parklands.png")

    Steppe = Image.open("worldengine/data/steppe.png")

    Chaparral = Image.open("worldengine/data/chaparral.png")

    Savanna = Image.open("worldengine/data/savanna.png")

    land_color = (
        181, 166, 127, 255)  # TODO: Put this in the argument list too??
    borders = _find_land_borders(world, resize_factor)

    if draw_outer_land_border:
        outer_borders = _find_outer_borders(world, resize_factor, borders)
        outer_borders = _find_outer_borders(world, resize_factor, outer_borders)

    if draw_mountains:  # TODO: numpy offers masked arrays - maybe they can be leveraged for all this?
        mountains_mask = _find_mountains_mask(world, resize_factor)
    if draw_biome:
        print("Creating masks")
        boreal_forest_mask = _createmask(world, world.is_boreal_forest, resize_factor)
        temperate_forest_mask = _createmask(world, world.is_temperate_forest, resize_factor)
        warm_temperate_forest_mask = \
            _createmask(world, world.is_warm_temperate_forest, resize_factor)
        tropical_dry_forest_mask = _createmask(world, world.is_tropical_dry_forest, resize_factor)
        jungle_mask = _createmask(world, world.is_jungle, resize_factor)
        cool_desert_mask = _createmask(world, world.is_cool_desert, resize_factor)
        hot_desert_mask = _createmask(world, world.is_hot_desert, resize_factor)
        rock_desert_mask = _createmask(world, world.is_hot_desert, resize_factor)  # TODO: add is_desert_mask
        ice_mask = _createmask(world, world.is_iceland, resize_factor)
        tundra_mask = _createmask(world, world.is_tundra, resize_factor)
        parklands_mask = _createmask(world, world.is_cold_parklands, resize_factor)
        steppe_mask = _createmask(world, world.is_steppe, resize_factor)
        chaparral_mask = _createmask(world, world.is_chaparral, resize_factor)
        savanna_mask = _createmask(world, world.is_savanna, resize_factor)
        
    def unset_mask(pos):
        x, y = pos
        mountains_mask[y, x] = 0

    def unset_boreal_forest_mask(pos):
        boreal_forest_mask.putpixel(pos,(0,0,0,0))

    def unset_temperate_forest_mask(pos):
        temperate_forest_mask.putpixel(pos,(0,0,0,0))

    def unset_warm_temperate_forest_mask(pos):
        warm_temperate_forest_mask.putpixel(pos,(0,0,0,0))

    def unset_tropical_dry_forest_mask(pos):
        tropical_dry_forest_mask.putpixel(pos,(0,0,0,0))

    def unset_jungle_mask(pos):
        jungle_mask.putpixel(pos,(0,0,0,0))

    def unset_tundra_mask(pos):
        tundra_mask.putpixel(pos,(0,0,0,0))

    def unset_savanna_mask(pos):
        savanna_mask.putpixel(pos,(0,0,0,0))

    def unset_hot_desert_mask(pos):
        hot_desert_mask.putpixel(pos,(0,0,0,0))

    def unset_rock_desert_mask(pos):
        rock_desert_mask.putpixel(pos,(0,0,0,0))

    def unset_cold_parklands_mask(pos):
        parklands_mask.putpixel(pos,(0,0,0,0))

    def unset_steppe_mask(pos):
        steppe_mask.putpixel(pos,(0,0,0,0))

    def unset_cool_desert_mask(pos):
        cool_desert_mask.putpixel(pos,(0,0,0,0))

    def unset_chaparral_mask(pos):
        chaparral_mask.putpixel(pos,(0,0,0,0))

    def on_border(pos):
        x, y = pos
        return borders[y, x]

    if draw_rivers:
        print("Cutting rivers from masks")
        for y in range(world.height):
            for x in range(world.width):
                if world.is_land((x, y)) and (world.river_map[x, y] > 0.0):
                    for dx in range(x*resize_factor-9,(x+1)*resize_factor+9):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+6):
                            unset_hot_desert_mask((dx, dy))
                    for dx in range(x*resize_factor-6,(x+1)*resize_factor+6):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+15):
                            unset_boreal_forest_mask((dx, dy))
                            unset_temperate_forest_mask((dx, dy))
                            unset_warm_temperate_forest_mask((dx, dy))
                            unset_tropical_dry_forest_mask((dx, dy))
                            unset_jungle_mask((dx, dy))
                    for dx in range(x*resize_factor,(x+1)*resize_factor):
                        for dy in range(y*resize_factor,(y+1)*resize_factor):
                            unset_tundra_mask((dx, dy))
                            unset_savanna_mask((dx, dy))
                            unset_rock_desert_mask((dx, dy))
                            unset_cold_parklands_mask((dx, dy))
                            unset_steppe_mask((dx, dy))
                            unset_cool_desert_mask((dx, dy))
                            unset_chaparral_mask((dx, dy))
                if world.is_land((x, y)) and (world.lake_map[x, y] != 0):
                    for dx in range(x*resize_factor-9,(x+1)*resize_factor+9):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+6):
                            unset_hot_desert_mask((dx, dy))
                    for dx in range(x*resize_factor-6,(x+1)*resize_factor+6):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+15):
                            unset_boreal_forest_mask((dx, dy))
                            unset_temperate_forest_mask((dx, dy))
                            unset_warm_temperate_forest_mask((dx, dy))
                            unset_tropical_dry_forest_mask((dx, dy))
                            unset_jungle_mask((dx, dy))
                    for dx in range(x*resize_factor,(x+1)*resize_factor):
                        for dy in range(y*resize_factor,(y+1)*resize_factor):
                            unset_tundra_mask((dx, dy))
                            unset_savanna_mask((dx, dy))
                            unset_rock_desert_mask((dx, dy))
                            unset_cold_parklands_mask((dx, dy))
                            unset_steppe_mask((dx, dy))
                            unset_cool_desert_mask((dx, dy))
                            unset_chaparral_mask((dx, dy))

    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...drawing_functions.draw_oldmap_on_pixel: init Elapsed time " +
            str(elapsed_time) + " seconds.")
        sys.stdout.flush()

    if verbose:
        start_time = time.time()
    border_color = (0, 0, 0, 255)
    outer_border_color = gradient(0.5, 0, 1.0, rgba_to_rgb(border_color), rgba_to_rgb(sea_color))
    for y in range(resize_factor * world.height):
        for x in range(resize_factor * world.width):
            xf = int(x / resize_factor)
            yf = int(y / resize_factor)
            if borders[y, x]:
                target.set_pixel(x, y, border_color)
            elif draw_outer_land_border and outer_borders[y, x]:
                target.set_pixel(x, y, outer_border_color)
            elif world.ocean[yf, xf]:
                target.set_pixel(x, y, sea_color)
            else:
                target.set_pixel(x, y, land_color)
    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...drawing_functions.draw_oldmap_on_pixel: color ocean " +
            "Elapsed time " + str(elapsed_time) + " seconds.")

    if verbose:
        start_time = time.time()

    def anti_alias(steps):

        def _anti_alias_step():
            for y in range(resize_factor * world.height):
                for x in range(resize_factor * world.width):
                    _anti_alias_point(x, y)

        def _anti_alias_point(x, y):
            n = 2
            tot_r = target[x, y][0] * 2
            tot_g = target[x, y][1] * 2
            tot_b = target[x, y][2] * 2
            for dy in range(-1, +2):
                py = y + dy
                if py > 0 and py < resize_factor * world.height:
                    for dx in range(-1, +2):
                        px = x + dx
                        if px > 0 and px < resize_factor * world.width:
                            n += 1
                            tot_r += target[px, py][0]
                            tot_g += target[px, py][1]
                            tot_b += target[px, py][2]
            r = int(tot_r / n)
            g = int(tot_g / n)
            b = int(tot_b / n)
            target[x, y] = (r, g, b, 255)

        for i in range(steps):
            _anti_alias_step()

    anti_alias(1)
    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...drawing_functions.draw_oldmap_on_pixel: anti alias " +
            "Elapsed time " + str(elapsed_time) + " seconds.")

    # Draw texture biomes
    if draw_biome:
        print("Drawing textures")
        _texture(target, Ice, ice_mask)
        _texture(target, Tundra, tundra_mask)
        _texture(target, Parklands, parklands_mask)
        _texture(target, Steppe, steppe_mask)
        _texture(target, Chaparral, chaparral_mask)
        _texture(target, Savanna, savanna_mask)

        # Draw cool desert
        print("Drawing cool deserts")
        for y in range(resize_factor * world.height):
            for x in range(resize_factor * world.width):
                pos = (x,y)
                if cool_desert_mask.getpixel(pos)[3] == 255:
                    w = 8
                    h = 2
                    r = 9
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _draw_cool_desert(target, x, y, w=w, h=h)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     action=unset_cool_desert_mask)

        # Draw icons
        print("Drawing icons")
        for y in range(resize_factor * world.height):
            for x in range(resize_factor * world.width):
                pos = (x,y)
                if hot_desert_mask.getpixel(pos)[3] == 255:
                    w = 6
                    h = 2
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        d = int(len(Deserts) * rng.random_sample())
                        _stamp(target, Deserts[d], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     action=unset_hot_desert_mask)
                        
                if boreal_forest_mask.getpixel(pos)[3] == 255:
                    w = 4
                    h = 5
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(Pines) * rng.random_sample())
                        _stamp(target, Pines[m], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_boreal_forest_mask)

                if temperate_forest_mask.getpixel(pos)[3] == 255:
                    w = 4
                    h = 5
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        if rng.random_sample() <= .5:
                            m = int(len(Pines) * rng.random_sample())
                            _stamp(target, Pines[m], x, y, w=w*3, h=h*3)
                        else:
                            m = int(len(Decids) * rng.random_sample())
                            _stamp(target, Decids[m], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_temperate_forest_mask)

                if warm_temperate_forest_mask.getpixel(pos)[3] == 255:
                    w = 4
                    h = 5
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(Decids) * rng.random_sample())
                        _stamp(target, Decids[m], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_warm_temperate_forest_mask)

                if tropical_dry_forest_mask.getpixel(pos)[3] == 255:
                    w = 4
                    h = 5
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(DryTropicals) * rng.random_sample())
                        _stamp(target, DryTropicals[m], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_tropical_dry_forest_mask)
                
                if jungle_mask.getpixel(pos)[3] == 255:
                    w = 4
                    h = 5
                    r = 6
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(Jungles) * rng.random_sample())
                        _stamp(target, Jungles[m], x, y, w=w*3, h=h*3)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     action=unset_jungle_mask)

    if draw_rivers:
        print("Drawing rivers")
        draw_rivers_on_image(world, target, resize_factor)

    # Draw mountains
    if draw_mountains:
        print("Drawing mountains")
        if verbose:
            start_time = time.time()
        for y in range(resize_factor * world.height):
            for x in range(resize_factor * world.width):
                if mountains_mask[y, x] > 8.5:
                    h = int(mountains_mask[y, x] * 2 / 3)
                    w = int(h * 3 / 2)
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(Mountains) * rng.random_sample())
                        _stamp(target, Mountains[m], x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)
                elif mountains_mask[y, x] > 0:
                    h = int(mountains_mask[y, x] * 2 / 3)
                    w = int(h * (4 / ( 1 + (h/3.4)) ))
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        
                        m = int(len(Hills) * rng.random_sample())
                        _stamp(target, Hills[m], x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)
        if verbose:
            elapsed_time = time.time() - start_time
            print(
                "...drawing_functions.draw_oldmap_on_pixel: draw mountains " +
                "Elapsed time " + str(elapsed_time) + " seconds.")
