# quicklens/notes/shts/plot_rec_graph.py
# --
# produces a figure illustrating the recursion relations used to
# determine the spherical harmonic coefficients _{s} Y_{lm}(\theta, \phi),
# as discussed in quicklens/notes/shts/shts.tex.

import copy
import numpy as n
from pyx import *

unit.set(xscale=1.25)

xmin = -8
xmax = 8
ymin = -2
ymax = 8

s    = 2

xmid = 0
ymid = 0

assert( xmax == ymax )
assert( xmin == -xmax )

class notexter():
    def labels(self, ticks):
        for t in ticks:
            t.label = ''

def tcirc(x,y,t):
    tn = c.text(x, y, t, [text.valign.middle, text.halign.boxcenter])
    ts = (tn.bbox().height() + tn.bbox().width())/2.0
    c.stroke(path.circle(x, y, ts))

def wedge(c, x, y, s, a):
    # x, y, size, angle
    return c.stroke( path.line(-1, 1, 0, 0) << path.line(0, 0, -1, -1), [trafo.scale(sx=s, sy=s), trafo.rotate(a), trafo.translate(x, y)] )

c = canvas.canvas()
c.insert( graph.axis.pathaxis(path.line(xmin,ymid,xmax,ymid), graph.axis.linear(min=xmin, max=xmax, texter=notexter())) )
c.insert( graph.axis.pathaxis(path.line(xmid,ymin,xmid,ymax), graph.axis.linear(min=ymin, max=ymax, texter=notexter())) )
c.stroke( path.line(xmin, 0, xmax, 0) )
c.stroke( path.line(0, ymin, 0, ymax) )
p1 = path.line(xmin, ymax, -s, s) << path.line(-s,s,s,s) << path.line(s,s, xmax, xmax)
c.stroke( p1 )

so = .5
c.stroke(path.line(s+.75, s, s+2.25, s), [style.linestyle.dashed])
c.fill(path.circle(s+1.5, s, .2),[color.rgb.white])
c.text(s+1.5, s, r"\Large{s}", [text.valign.middle, text.halign.boxcenter])

tm = c.text(.2, -1.5, r"\Large{$\ell$}", [text.valign.bottom, text.valign.top])
tl = c.text(7, -.8, r"\Large{m}", [text.halign.boxleft])

ws = .2
wedge(c, -2.5, 2.5, ws, 135)
wedge(c,-4.5, 4.5, ws, 135)
wedge(c,-6.5, 6.5, ws, 135)

wedge(c, 1.5, s, ws, 0)
wedge(c, -1.5, s, ws, 180)

wedge(c, 2.5, 2.5, ws, 45)
wedge(c, 4.5, 4.5, ws, 45)
wedge(c, 6.5, 6.5, ws, 45)

wedge(c, 0, 1, ws, 90)
tcirc(-.7,1,'A')

tcirc(1, 1.5,'B')
tcirc(5.0, 4.0, 'C')

p2 = path.line(xmin, ymax, -s, s) << path.line(-s,s,s,s) << path.line(s,s, xmax, xmax)
p2.append(path.closepath())
c2 = canvas.canvas([canvas.clip(p2)])
for i in n.arange(xmin, xmax):
    c2.stroke( path.line(i, max(s,abs(i)), i, ymax+max(s,abs(i))))

c.insert(c2)

c.writePDFfile("rec_graph")
