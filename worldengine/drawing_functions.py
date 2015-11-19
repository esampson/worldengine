"""
This file should contain only functions that operates on pixels, not on images,
so no references to PIL are necessary and the module can be used also through
Jython
"""

import numpy
import sys
import time
from PIL import ImageFilter
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
            if world.is_land((x, y)) and (world.river_map[y, x] > 0.0):
                for dx in range(factor):
                    for dy in range(factor):
                        target.set_pixel(x * factor + dx, y * factor + dy, (0, 0, 128, 255))
            if world.is_land((x, y)) and (world.lake_map[y, x] != 0):
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

<<<<<<< HEAD
def _createmask(world, predicate, factor, odds=1):
    print(odds)
    rng = numpy.random.RandomState(world.seed)
    width = world.width * factor
    height = world.height * factor
    _mask = Image.new("RGBA", (width,height))
    for y in range(height):
        for x in range(width):
            xf = int(x / factor)
            yf = int(y / factor)
            if predicate((xf, yf)) and rng.random_sample() < odds:
                v = len(
                    world.tiles_around((xf, yf), radius=1,
                                       predicate=world.is_land))
                if v > 5:
                    _mask.putpixel((x,y),(0,0,0,255))
    return _mask  

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

class Bitmap:
    #self.h = None        #Height of area 'occupied' in map
    #self.w = None        #Width of area 'occupied' in map
    #self.r = None        #Area removed from mask
    #self.ph = None       #Height of printed bitmap
    #self.pw = None       #Width of printed bitmap
    #self.img = None      #Bitmap image

    def __init__(self, space, pw, ph, img):
        self.space = space
        self.ph = ph
        self.pw = pw
        self.img = img
    
class MapComponent:

    def __init__(self):
        self.Type = None      #Valid types should be ICONS, TEXTURE, BOTH, or None
        self.Bitmap = []

    def add_bitmap(self, bitmap):
        self.Bitmap.append(bitmap)

    def set_type(self, T):
        self.Type = T

def draw_ancientmap(world, target, resize_factor=1,
                    sea_color=(212, 198, 169, 255),
                    draw_biome = True, draw_rivers = True, draw_mountains = True,
                    draw_outer_land_border = False, verbose = False):
    rng = numpy.random.RandomState(world.seed)  # create our own random generator

    if verbose:
        start_time = time.time()
    print("Loading bitmaps")

    Icons = Image.open("worldengine/data/Icons.png")

    Mountain = MapComponent()
    Mountain.add_bitmap(Bitmap(0,0,0,Icons.crop((4,259,53,298))))
    Mountain.add_bitmap(Bitmap(0,0,0,Icons.crop((56,259,105,298))))
    Mountain.add_bitmap(Bitmap(0,0,0,Icons.crop((108,259,157,298))))
    Mountain.add_bitmap(Bitmap(0,0,0,Icons.crop((4,301,53,340))))
    Mountain.add_bitmap(Bitmap(0,0,0,Icons.crop((56,301,105,340))))
    
    SnowMountain = MapComponent()
    SnowMountain.add_bitmap(Bitmap(0,0,0,Icons.crop((160,259,209,298))))
    SnowMountain.add_bitmap(Bitmap(0,0,0,Icons.crop((212,259,261,298))))
    SnowMountain.add_bitmap(Bitmap(0,0,0,Icons.crop((264,259,313,298))))
    SnowMountain.add_bitmap(Bitmap(0,0,0,Icons.crop((160,301,209,340))))
    SnowMountain.add_bitmap(Bitmap(0,0,0,Icons.crop((212,301,261,340))))

    Hill = MapComponent()
    Hill.add_bitmap(Bitmap(0,0,0,Icons.crop((4,343,103,367))))
    Hill.add_bitmap(Bitmap(0,0,0,Icons.crop((4,370,103,394))))
    Hill.add_bitmap(Bitmap(0,0,0,Icons.crop((4,397,103,421))))

    SnowHill = MapComponent()
    SnowHill.add_bitmap(Bitmap(0,0,0,Icons.crop((370,343,469,367))))
    SnowHill.add_bitmap(Bitmap(0,0,0,Icons.crop((370,370,469,394))))
    SnowHill.add_bitmap(Bitmap(0,0,0,Icons.crop((370,397,469,421))))
    
    HotDesert = MapComponent()
    HotDesert.add_bitmap(Bitmap(9,27,9,Icons.crop((33,70,59,78))))
    HotDesert.add_bitmap(Bitmap(9,27,9,Icons.crop((62,70,88,78))))
    HotDesert.add_bitmap(Bitmap(9,27,9,Icons.crop((91,70,117,78))))

    Decid = MapComponent()
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((4,4,15,18))))
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((18,4,29,18))))
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((32,4,43,18))))
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((46,4,57,18))))
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((60,4,71,18))))
    Decid.add_bitmap(Bitmap(6,12,15,Icons.crop((74,4,85,18))))

    Pine = MapComponent()
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((4,38,15,52))))
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((18,38,29,52))))
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((32,38,43,52))))
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((46,38,57,52))))
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((60,38,71,52))))
    Pine.add_bitmap(Bitmap(5,12,15,Icons.crop((74,38,85,52))))

    Mixed = MapComponent()
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((4,4,15,18))))
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((18,4,29,18))))
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((32,4,43,18))))
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((46,4,57,18))))
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((60,4,71,18))))
    Mixed.add_bitmap(Bitmap(6,12,15,Icons.crop((74,4,85,18))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((4,38,15,52))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((18,38,29,52))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((32,38,43,52))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((46,38,57,52))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((60,38,71,52))))
    Mixed.add_bitmap(Bitmap(5,12,15,Icons.crop((74,38,85,52))))

    Jungle = MapComponent()
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((4,21,15,35))))
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((18,21,29,35))))
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((32,21,43,35))))
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((46,21,57,35))))
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((60,21,71,35))))
    Jungle.add_bitmap(Bitmap(6,12,15,Icons.crop((74,21,85,35))))

    DryTropical = MapComponent()
    DryTropical.add_bitmap(Bitmap(6,12,15,Icons.crop((4,55,15,69))))
    DryTropical.add_bitmap(Bitmap(6,12,15,Icons.crop((18,55,29,69))))

    Ice = MapComponent()
    Ice.add_bitmap(Bitmap(0,0,0,Icons.crop((4,80,43,119))))

    Tundra = MapComponent()
    Tundra.add_bitmap(Bitmap(0,0,0,Icons.crop((46,122,85,161))))

    Parkland = MapComponent()
    Parkland.add_bitmap(Bitmap(0,0,0,Icons.crop((46,80,85,119))))

    Steppe = MapComponent()
    Steppe.add_bitmap(Bitmap(0,0,0,Icons.crop((4,122,43,161))))

    Chaparral = MapComponent()
    Chaparral.add_bitmap(Bitmap(0,0,0,Icons.crop((4,164,83,196))))

    Savanna = MapComponent()
    Savanna.add_bitmap(Bitmap(0,0,0,Icons.crop((88,80,127,119))))

    CoolDesert = MapComponent()
    CoolDesert.add_bitmap(Bitmap(0,0,0,Icons.crop((4,199,83,256))))

    Water = MapComponent()
    Water.add_bitmap(Bitmap(0,0,0,Icons.crop((130, 4, 329, 203))))

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
        land_mask = boreal_forest_mask = _createmask(world, world.is_land, resize_factor, 1)
        water_mask = land_mask.filter(ImageFilter.GaussianBlur(16))
        boreal_forest_mask = _createmask(world, world.is_boreal_forest, resize_factor, .25)
        temperate_forest_mask = _createmask(world, world.is_temperate_forest, resize_factor, .25)
        warm_temperate_forest_mask = \
            _createmask(world, world.is_warm_temperate_forest, resize_factor, .25)
        tropical_dry_forest_mask = _createmask(world, world.is_tropical_dry_forest, resize_factor, .25)
        jungle_mask = _createmask(world, world.is_jungle, resize_factor, .25)
        cool_desert_mask = _createmask(world, world.is_cool_desert, resize_factor, 1)
        hot_desert_mask = _createmask(world, world.is_hot_desert, resize_factor, .41667)
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
                    for dx in range(x*resize_factor-12,(x+1)*resize_factor+12):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+8):
                            unset_hot_desert_mask((dx, dy))
                    for dx in range(x*resize_factor-5,(x+1)*resize_factor+5):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+14):
                            unset_boreal_forest_mask((dx, dy))
                            unset_temperate_forest_mask((dx, dy))
                            unset_warm_temperate_forest_mask((dx, dy))
                            unset_tropical_dry_forest_mask((dx, dy))
                            unset_jungle_mask((dx, dy))
                    for dx in range(x*resize_factor,(x+1)*resize_factor):
                        for dy in range(y*resize_factor,(y+1)*resize_factor):
                            unset_tundra_mask((dx, dy))
                            unset_savanna_mask((dx, dy))
                            unset_cold_parklands_mask((dx, dy))
                            unset_steppe_mask((dx, dy))
                            unset_chaparral_mask((dx, dy))
                            unset_cool_desert_mask((dx, dy))
                if world.is_land((x, y)) and (world.lake_map[x, y] != 0):
                    for dx in range(x*resize_factor-12,(x+1)*resize_factor+12):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+8):
                            unset_hot_desert_mask((dx, dy))
                    for dx in range(x*resize_factor-5,(x+1)*resize_factor+5):
                        for dy in range(y*resize_factor,(y+1)*resize_factor+14):
                            unset_boreal_forest_mask((dx, dy))
                            unset_temperate_forest_mask((dx, dy))
                            unset_warm_temperate_forest_mask((dx, dy))
                            unset_tropical_dry_forest_mask((dx, dy))
                            unset_jungle_mask((dx, dy))
                    for dx in range(x*resize_factor,(x+1)*resize_factor):
                        for dy in range(y*resize_factor,(y+1)*resize_factor):
                            unset_tundra_mask((dx, dy))
                            unset_savanna_mask((dx, dy))
                            unset_cold_parklands_mask((dx, dy))
                            unset_steppe_mask((dx, dy))
                            unset_chaparral_mask((dx, dy))
                            unset_cool_desert_mask((dx, dy))

    if verbose:
        elapsed_time = time.time() - start_time
        print(
            "...drawing_functions.draw_oldmap_on_pixel: init Elapsed time " +
            str(elapsed_time) + " seconds.")
        sys.stdout.flush()

    for y in range(resize_factor * world.height):
        for x in range(resize_factor * world.width):
            target.set_pixel(x, y, sea_color)

    _texture(target, Water.Bitmap[0].img, water_mask)
    
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
            elif not world.ocean[yf, xf]:
                target.set_pixel(x, y, land_color)
            #else:
            #    target.set_pixel(x, y, land_color)
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
            tot_r = target[y, x][0] * 2
            tot_g = target[y, x][1] * 2
            tot_b = target[y, x][2] * 2
            for dy in range(-1, +2):
                py = y + dy
                if py > 0 and py < resize_factor * world.height:
                    for dx in range(-1, +2):
                        px = x + dx
                        if px > 0 and px < resize_factor * world.width:
                            n += 1
                            tot_r += target[py, px][0]
                            tot_g += target[py, px][1]
                            tot_b += target[py, px][2]
            r = int(tot_r / n)
            g = int(tot_g / n)
            b = int(tot_b / n)
            target[y, x] = (r, g, b, 255)

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
        _texture(target, Ice.Bitmap[0].img, ice_mask)
        _texture(target, Tundra.Bitmap[0].img, tundra_mask)
        _texture(target, Parkland.Bitmap[0].img, parklands_mask)
        _texture(target, Steppe.Bitmap[0].img, steppe_mask)
        _texture(target, Chaparral.Bitmap[0].img, chaparral_mask)
        _texture(target, Savanna.Bitmap[0].img, savanna_mask)
        _texture(target, CoolDesert.Bitmap[0].img, cool_desert_mask)
             
        # Draw icons
        print("Drawing icons")
        for y in range(resize_factor * world.height):
            for x in range(resize_factor * world.width):
                pos = (x, y)
                fx = int(x/resize_factor)
                fy = int(y/resize_factor)
                
                if hot_desert_mask.getpixel(pos)[3] == 255:
                    m = int(len(HotDesert.Bitmap) * rng.random_sample())
                    r = HotDesert.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _stamp(target, HotDesert.Bitmap[m].img, x, y, w=HotDesert.Bitmap[m].pw, h=HotDesert.Bitmap[m].ph)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     action=unset_hot_desert_mask)
                        
                if boreal_forest_mask.getpixel(pos)[3] == 255:
                    m = int(len(Pine.Bitmap) * rng.random_sample())
                    r = Pine.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        
                        _stamp(target, Pine.Bitmap[m].img, x, y, w=Pine.Bitmap[m].pw, h=Pine.Bitmap[m].ph)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_boreal_forest_mask)

                if temperate_forest_mask.getpixel(pos)[3] == 255:
                    m = int(len(Mixed.Bitmap) * rng.random_sample())
                    r = Mixed.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _stamp(target, Mixed.Bitmap[m].img, x, y, w=Mixed.Bitmap[m].pw, h=Mixed.Bitmap[m].ph)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_temperate_forest_mask)

                if warm_temperate_forest_mask.getpixel(pos)[3] == 255:
                    m = int(len(Decid.Bitmap) * rng.random_sample())
                    r = Decid.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _stamp(target, Decid.Bitmap[m].img, x, y, w=Decid.Bitmap[m].pw, h=Decid.Bitmap[m].ph)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_warm_temperate_forest_mask)

                if tropical_dry_forest_mask.getpixel(pos)[3] == 255:
                    m = int(len(DryTropical.Bitmap) * rng.random_sample())
                    r = DryTropical.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _stamp(target, DryTropical.Bitmap[m].img, x, y, w=DryTropical.Bitmap[m].pw, h=DryTropical.Bitmap[m].ph)
                        world.on_tiles_around_factor(
                            resize_factor, (x, y),
                            radius=r,
                            action=unset_tropical_dry_forest_mask)
                
                if jungle_mask.getpixel(pos)[3] == 255:
                    m = int(len(Jungle.Bitmap) * rng.random_sample())
                    r = Jungle.Bitmap[m].space
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        _stamp(target, Jungle.Bitmap[m].img, x, y, w=Jungle.Bitmap[m].pw, h=Jungle.Bitmap[m].ph)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     action=unset_jungle_mask)

                if mountains_mask[y, x] > 8.5 and not world.is_temperature_polar((fx,fy)):
                    h = int(mountains_mask[y, x] * 2 / 3) * 2
                    w = int(h * 3 / 2)
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(Mountain.Bitmap) * rng.random_sample())
                        _stamp(target, Mountain.Bitmap[m].img, x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)
                if mountains_mask[y, x] > 0 and mountains_mask[y, x] <= 8.5 and not world.is_temperature_polar((fx,fy)):
                    h = int(mountains_mask[y, x] * 2 / 3) * 2
                    w = int(h * (4 / ( 1 + (h/3.4)) ))
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        
                        m = int(len(Hill.Bitmap) * rng.random_sample())
                        _stamp(target, Hill.Bitmap[m].img, x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)
                        
                if mountains_mask[y, x] > 8.5 and world.is_temperature_polar((fx,fy)):
                    h = int(mountains_mask[y, x] * 2 / 3) * 2
                    w = int(h * 3 / 2)
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        m = int(len(SnowMountain.Bitmap) * rng.random_sample())
                        _stamp(target, SnowMountain.Bitmap[m].img, x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)
                if mountains_mask[y, x] > 0 and mountains_mask[y, x] <= 8.5 and world.is_temperature_polar((fx,fy)):
                    h = int(mountains_mask[y, x] * 2 / 3) * 2
                    w = int(h * (4 / ( 1 + (h/3.4)) ))
                    r = max(int(w / 3 * 2), h)
                    if len(world.tiles_around_factor(resize_factor, (x, y),
                                                     radius=r,
                                                     predicate=on_border)) <= 2:
                        
                        m = int(len(SnowHill.Bitmap) * rng.random_sample())
                        _stamp(target, SnowHill.Bitmap[m].img, x, y, w=w*2, h=h*2)
                        world.on_tiles_around_factor(resize_factor, (x, y),
                                                     radius=r, action=unset_mask)

                if world.is_land((fx, fy)) and (world.river_map[fx, fy] > 0.0):
                    target.set_pixel(x, y, (0, 0, 128, 255))
                if world.is_land((fx, fy)) and (world.lake_map[fx, fy] != 0):
                    target.set_pixel(x, y, (0, 100, 128, 255))

    #if draw_rivers:
    #    print("Drawing rivers")
    #    draw_rivers_on_image(world, target, resize_factor)
        
