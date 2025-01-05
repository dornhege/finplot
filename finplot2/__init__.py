from math import ceil, floor
from typing import Optional
from collections import defaultdict
from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from PyQt5.QtCore import Qt, QEvent

print("LOADING FINPLOT 2")

# The default theme used to created plots
theme = {
    "foreground": "#000",
    "background": "#fff",
    "odd_plot_background": "#eaeaea",
}

# axis_height_factor determines the y-stretch for the n-th axis object in a
# FinWindow. When resizing the n-th axis is displayed a factor of
# axis_height_factor[n] larger than the others. A default is 1 for each axis.
axis_height_factor = defaultdict(lambda: 1)
axis_height_factor.update({0: 2})


timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
display_timezone = tzlocal()  # default to local
truncate_timestamp = True
winx, winy, winw, winh = 300, 150, 800, 400
win_recreate_delta = 30
# format: mode, min-duration, pd-freq-fmt, tick-str-len
time_splits = [
    ("years", 2 * 365 * 24 * 60 * 60, "YS", 4),
    ("months", 3 * 30 * 24 * 60 * 60, "MS", 10),
    ("weeks", 3 * 7 * 24 * 60 * 60, "W-MON", 10),
    ("days", 3 * 24 * 60 * 60, "D", 10),
    ("hours", 9 * 60 * 60, "3h", 16),
    ("hours", 3 * 60 * 60, "h", 16),
    ("minutes", 45 * 60, "15min", 16),
    ("minutes", 15 * 60, "5min", 16),
    ("minutes", 3 * 60, "min", 16),
    ("seconds", 45, "15s", 19),
    ("seconds", 15, "5s", 19),
    ("seconds", 3, "s", 19),
    ("milliseconds", 0, "ms", 23),
]
epoch_period = 1e30

# All FinWindow instances created from this module
windows = []

# TODO
# Create FinPlotItem(pg.PlotItem) to avoid monkey patching the functions in?
# - Would clash with the FinPlotItem that isn't a PlotItem in FinPlot.
# Goals:
# 1. Open a window/viewbox and check zooming, etc.
# 2. Implement one simple plotting function (candlestick or plot default)
#     Maybe plot default iff that doesn't require a datasrc
# Can we fix the missing functions from parents in LSP - due to hacks of importing them down?
# Then: Compare/add missing functionality from finplot. First for basic charts


class FinWindow(pg.GraphicsLayoutWidget):
    closing: bool

    def __init__(self, title: str, **kwargs):
        global winx, winy
        pg.mkQApp()  # Super() does this. Why do we here and also in show()
        super().__init__(**kwargs)  # TODO this creates a qApp. Catch that
        self.setWindowTitle(title)
        self.setGeometry(winx, winy, winw, winh)
        winx = (winx + win_recreate_delta) % 800
        winy = (winy + win_recreate_delta) % 500
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(-1)

        self.closing = False

    @property
    def axs(self) -> list[pg.PlotItem]:
        """Returns all PlotItems within this window."""
        return [ax for ax in self.ci.items if isinstance(ax, pg.PlotItem)]

    def close(self) -> None:
        """
        Overrides GraphicsView.close
        """
        self.closing = True
        return super().close()

    def resizeEvent(self, ev: Optional[QtGui.QResizeEvent]) -> None:
        """
        We resize and set the Y axis scale factor according to the axis_height_factor.
        No point in trying to use the "row stretch factor" in Qt which is broken
        beyond repair.

        Overrides QWidget.resizeEvent
        """
        if ev and not self.closing:
            new_win_height = ev.size().height()
            old_win_height = (
                ev.oldSize().height() if ev.oldSize().height() > 0 else new_win_height
            )
            client_borders = old_win_height - sum(
                ax.vb.size().height() for ax in self.axs if ax.vb is not None
            )
            client_borders = min(max(client_borders, 0), 30)  # clamp
            new_axes_height = new_win_height - client_borders
            axis_height_factor_total = 0.0
            for i in range(len(self.axs)):
                axis_height_factor_total += axis_height_factor[i]
            for i, ax in enumerate(self.axs):
                f = axis_height_factor[i] / axis_height_factor_total
                ax.setMinimumSize(
                    100 if axis_height_factor[i] > 1 else 50, new_axes_height * f
                )
        return super().resizeEvent(ev)

    def leaveEvent(self, ev: Optional[QtCore.QEvent]) -> None:
        """
        Overrides QWidget.leaveEvent
        """
        if not self.closing:
            super().leaveEvent(ev)


class FinViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.setMouseEnabled(x=True, y=True)
        self.setRange(QtCore.QRectF(pg.Point(0, 0), pg.Point(1, 1)))


# Check DataSrc and how it deals with times/timestamps/indices:
# This might boil down to understanding how the x-axis works,
# how items are inserted (by x-axis position or by index => Gap issue)
# what an x-axis entry even is (index, timestamp, float [s]?)
# or just how an x-axis item/tick is shown as a string
#
# Keep for now:
# They probably shouldn't exist or be factored out somewhere. Ideally covered by standard python/pandas functions.
# If these functions are really needed (and understood why), maybe factor out into a module
# with 1-2 public functions. No need to spam this main module


def _millisecond_tz_wrap(s):
    if len(s) > 6 and s[-6] in "+-" and s[-3] == ":":  # +01:00 fmt timezone present?
        s = s[:-6]
    return (s + ".000000") if "." not in s else s


def _x2local_t(datasrc, x):
    if display_timezone == None:
        return _x2utc(datasrc, x)
    return _x2t(
        datasrc,
        x,
        lambda t: _millisecond_tz_wrap(
            datetime.fromtimestamp(t / 1e9, tz=display_timezone).strftime(
                timestamp_format
            )
        ),
    )


def _x2utc(datasrc, x):
    # using pd.to_datetime allow for pre-1970 dates
    return _x2t(
        datasrc, x, lambda t: pd.to_datetime(t, unit="ns").strftime(timestamp_format)
    )


def _x2t(datasrc, x, ts2str):
    if not datasrc:
        return "", False
    try:
        x += 0.5
        t, _, _, _, cnt = datasrc.hilo(x, x)
        if cnt:
            if not datasrc.timebased():
                return "%g" % t, False
            s = ts2str(t)
            if not truncate_timestamp:
                return s, True
            if epoch_period >= 23 * 60 * 60:  # daylight savings, leap seconds, etc
                i = s.index(" ")
            elif epoch_period >= 59:  # consider leap seconds
                i = s.rindex(":")
            elif epoch_period >= 1:
                i = s.index(".") if "." in s else len(s)
            elif epoch_period >= 0.001:
                i = -3
            else:
                i = len(s)
            return s[:i], True
    except Exception as e:
        import traceback

        traceback.print_exc()
    return "", datasrc.timebased()


def _x2year(datasrc, x):
    t, hasds = _x2local_t(datasrc, x)
    return t[:4], hasds


def _pdtime2epoch(t):
    if isinstance(t, pd.Series):
        if isinstance(t.iloc[0], pd.Timestamp):
            dtype = str(t.dtype)
            if dtype.endswith("[s]"):
                return t.astype("int64") * int(1e9)
            elif dtype.endswith("[ms]"):
                return t.astype("int64") * int(1e6)
            elif dtype.endswith("us"):
                return t.astype("int64") * int(1e3)
            return t.astype("int64")
        h = np.nanmax(t.values)
        if h < 1e10:  # handle s epochs
            return (t * 1e9).astype("int64")
        if h < 1e13:  # handle ms epochs
            return (t * 1e6).astype("int64")
        if h < 1e16:  # handle us epochs
            return (t * 1e3).astype("int64")
        return t.astype("int64")
    return t


lerp = lambda t, a, b: t * b + (1 - t) * a


def _pdtime2index(ax, ts, any_end=False, require_time=False):
    if isinstance(ts.iloc[0], pd.Timestamp):
        ts = ts.astype("int64")
    else:
        h = np.nanmax(ts.values)
        if h < 1e7:
            if require_time:
                assert False, "not a time series"
            return ts
        if h < 1e10:  # handle s epochs
            ts = ts.astype("float64") * 1e9
        elif h < 1e13:  # handle ms epochs
            ts = ts.astype("float64") * 1e6
        elif h < 1e16:  # handle us epochs
            ts = ts.astype("float64") * 1e3

    # datasrc = _get_datasrc(ax)
    datasrc = None
    xs = datasrc.x

    # try exact match before approximate match
    if all(xs.isin(ts)):
        exact = datasrc.index[ts].to_list()
        if len(exact) == len(ts):
            return exact

    r = []
    for i, t in enumerate(ts):
        xss = xs.loc[xs > t]
        if len(xss) == 0:
            t0 = xs.iloc[-1]
            if any_end or t0 == t:
                r.append(len(xs) - 1)
                continue
            if i > 0:
                continue
            assert t <= t0, "must plot this primitive in prior time-range"
        i1 = xss.index[0]
        i0 = i1 - 1
        if i0 < 0:
            i0, i1 = 0, 1
        t0, t1 = xs.loc[i0], xs.loc[i1]
        if t0 == t1:
            r.append(i0)
        else:
            dt = (t - t0) / (t1 - t0)
            r.append(lerp(dt, i0, i1))
    return r


def _is_str_midnight(s):
    return s.endswith(" 00:00") or s.endswith(" 00:00:00")


def _makepen(color, style=None, width=1):
    if style is None or style == "-":
        return pg.mkPen(color=color, width=width)
    dash = []
    for ch in style:
        if ch == "-":
            dash += [4, 2]
        elif ch == "_":
            dash += [10, 2]
        elif ch == ".":
            dash += [1, 2]
        elif ch == " ":
            if dash:
                dash[-1] += 2
    return pg.mkPen(
        color=color, style=QtCore.Qt.PenStyle.CustomDashLine, dash=dash, width=width
    )


class EpochAxisItem(pg.AxisItem):
    vb: FinViewBox

    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb

    def tickStrings(self, values, scale, spacing):
        """
        Overrides AxisItem.tickStrings
        """
        if self.mode == "num":
            return ["%g" % v for v in values]
        conv = _x2year if self.mode == "years" else _x2local_t
        strs = [conv(self.vb.datasrc, value)[0] for value in values]
        if all(
            _is_str_midnight(s) for s in strs if s
        ):  # all at midnight -> round to days
            strs = [s.partition(" ")[0] for s in strs]
        return strs

    def tickValues(self, minVal, maxVal, size):
        """
        Overrides AxisItem.tickValues
        """
        self.mode = "num"
        ax = self.vb.parent()
        # datasrc = _get_datasrc(ax, require=False)
        datasrc = None
        if datasrc is None or not self.vb.x_indexed:
            return super().tickValues(minVal, maxVal, size)
        # calculate if we use years, days, etc.
        t0, t1, _, _, _ = datasrc.hilo(minVal, maxVal)
        t0, t1 = pd.to_datetime(t0), pd.to_datetime(t1)
        dts = (t1 - t0).total_seconds()
        gfx_width = int(size)
        for mode, dtt, freq, ticklen in time_splits:
            if dts > dtt:
                self.mode = mode
                desired_ticks = (
                    gfx_width / ((ticklen + 2) * 10) - 1
                )  # an approximation is fine
                if self.vb.datasrc is not None and not self.vb.datasrc.is_smooth_time():
                    desired_ticks -= 1  # leave more space for unevenly spaced ticks
                desired_ticks = max(desired_ticks, 4)
                to_midnight = freq in ("YS", "MS", "W-MON", "D")
                tz = (
                    display_timezone if to_midnight else None
                )  # for shorter timeframes, timezone seems buggy
                rng = pd.date_range(t0, t1, tz=tz, normalize=to_midnight, freq=freq)
                steps = (
                    len(rng) if len(rng) & 1 == 0 else len(rng) + 1
                )  # reduce jitter between e.g. 5<-->10 ticks for resolution close to limit
                step = int(steps / desired_ticks) or 1
                rng = rng[::step]
                if not to_midnight:
                    try:
                        rng = rng.round(freq=freq)
                    except:
                        pass
                ax = self.vb.parent()
                rng = _pdtime2index(ax=ax, ts=pd.Series(rng), require_time=True)
                indices = [ceil(i) for i in rng if i > -1e200]
                return [(0, indices)]
        return [(0, [])]

    def generateDrawSpecs(self, p):
        """
        Overrides AxisItem.generateDrawSpecs
        """
        specs = super().generateDrawSpecs(p)
        if specs:
            if not self.style["showValues"]:
                pen, p0, p1 = specs[0]  # axis specs
                specs = [(_makepen("#fff0"), p0, p1)] + list(
                    specs[1:]
                )  # don't draw axis if hiding values
            else:
                # throw out ticks that are out of bounds
                text_specs = specs[2]
                if len(text_specs) >= 4:
                    rect, flags, text = text_specs[0]
                    if rect.left() < 0:
                        del text_specs[0]
                    rect, flags, text = text_specs[-1]
                    if rect.right() > self.geometry().width():
                        del text_specs[-1]
                # ... and those that overlap
                x = 1e6
                for i, (rect, flags, text) in reversed(list(enumerate(text_specs))):
                    if rect.right() >= x:
                        del text_specs[i]
                    else:
                        x = rect.left()
        return specs


class YAxisItem(pg.AxisItem):
    vb: FinViewBox
    hide_strings: bool
    next_fmt: str

    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb
        self.style["autoExpandTextSpace"] = False
        self.style["autoReduceTextSpace"] = False
        self.hide_strings = False
        self.next_fmt = "%g"

    def tickValues(self, minVal, maxVal, size):
        """
        Overrides AxisItem.tickValues
        """
        vs = super().tickValues(minVal, maxVal, size)
        if len(vs) < 3:
            return vs
        return self._fmt_values(vs)

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        """
        Overrides AxisItem.logTickValues
        """
        v1 = int(floor(minVal))
        v2 = int(ceil(maxVal))
        minor = []
        for v in range(v1, v2):
            minor.extend([v + l for l in np.log10(np.linspace(1, 9.9, 90))])
        minor = [x for x in minor if x > minVal and x < maxVal]
        if not minor:
            minor.extend(np.geomspace(minVal, maxVal, 7)[1:-1])
        if len(minor) > 10:
            minor = minor[:: len(minor) // 5]
        vs = [(None, minor)]
        return self._fmt_values(vs)

    def tickStrings(self, values, scale, spacing):
        """
        Overrides AxisItem.tickStrings
        """
        if self.hide_strings:
            return []
        xform = self.vb.yscale.xform
        return [self.next_fmt % xform(value) for value in values]

    def _fmt_values(self, vs):
        xform = self.vb.yscale.xform
        gs = ["%g" % xform(v) for v in vs[-1][1]]
        if not gs:
            return vs
        if any(["e" in g for g in gs]):
            maxdec = max(
                [len((g).partition(".")[2].partition("e")[0]) for g in gs if "e" in g]
            )
            self.next_fmt = "%%.%ie" % maxdec
        elif gs:
            maxdec = max([len((g).partition(".")[2]) for g in gs])
            self.next_fmt = "%%.%if" % maxdec
        else:
            self.next_fmt = "%g"
        return vs


def candlestick_ochl(
    datasrc, draw_body=True, draw_shadow=True, candle_width=0.6, ax=None
):
    pass


#    ax = _create_plot(ax=ax, maximize=False)
#    datasrc = _create_datasrc(ax, datasrc, ncols=5)
#    datasrc.scale_cols = [3,4] # only hi+lo scales
#    _set_datasrc(ax, datasrc)
#    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=draw_body, draw_shadow=draw_shadow, candle_width=candle_width, colorfunc=colorfunc, resamp='hilo')
#    _update_significants(ax, datasrc, force=True)
#    item.update_data = partial(_update_data, None, None, item)
#    item.update_gfx = partial(_update_gfx, item)
#    ax.addItem(item)
#    return item


def create_plot(title="Finance Plot", rows=1) -> list[pg.PlotItem]:
    """Creates a new FinWindow with `rows` PlotItems in it."""
    pg.setConfigOptions(foreground=theme["foreground"], background=theme["background"])
    window = FinWindow(title)
    windows.append(window)
    axs = _create_plot_items(rows=rows)
    for ax in axs:
        # ax = window.addPlot(title="Basic array plotting", y=np.random.normal(size=100))
        window.addItem(ax, col=1)
        window.nextRow()
    return axs


def _create_plot_items(rows=1) -> list[pg.PlotItem]:
    """Creates `rows` PlotItems.

    All PlotItems are created with a FinViewBox and if more than 1 PlotItem is
    created all items x-axis are linked to the first.
    """
    axs = []
    prev_ax = None
    for n in range(rows):
        viewbox = FinViewBox(enableMenu=False)
        ax = _create_timestamp_plot(prev_ax=prev_ax, viewbox=viewbox, index=n)
        if axs:
            assert ax.vb is not None
            ax.vb.setXLink(axs[0].vb)
        else:
            viewbox.setFocus()
        axs.append(ax)
        prev_ax = ax
    return axs


def show(qt_exec=True) -> None:
    """Shows all known FinWindows.

    If qt_exec is True, calls app.exec() on the current application.
    """
    print("Showing Windows")
    for win in windows:
        if isinstance(win, FinWindow) or qt_exec:
            win.show()
    if windows and qt_exec:
        app = QtGui.QGuiApplication.instance()
        assert app is not None
        app.exec()
        windows.clear()


def _create_timestamp_plot(
    prev_ax: pg.PlotItem | None, viewbox: FinViewBox, index: int
) -> pg.PlotItem:
    """Creates a PlotItem with the given `viewbox`."""
    if prev_ax is not None:
        print(f"SV, {type(prev_ax)=} {hasattr(prev_ax, 'set_visible')=}")
        # prev_ax.set_visible(xaxis=False) # hide the whole previous axis
    axes = {
        "bottom": EpochAxisItem(vb=viewbox, orientation="bottom"),
        "right": YAxisItem(vb=viewbox, orientation="right"),
    }
    ax = pg.PlotItem(
        viewBox=viewbox, axisItems=axes, name="plot-%i" % index, enableMenu=False
    )
    ax.setClipToView(True)
    ax.setDownsampling(auto=True, mode="subsample")
    ax.hideAxis("left")
    # if y_label_width:
    #    ax.axes['right']['item'].setWidth(y_label_width) # this is to put all graphs on equal footing when texts vary from 0.4 to 2000000
    # ax.axes['right']['item'].setStyle(tickLength=-5) # some bug, totally unexplicable (why setting the default value again would fix repaint width as axis scale down)
    # ax.axes['right']['item'].setZValue(30) # put axis in front instead of behind data
    # ax.axes['bottom']['item'].setZValue(30)
    # ax.significant_forced = False
    # ax.significant_decimals = significant_decimals
    # ax.significant_eps = significant_eps
    # ax.inverted = False
    # ax.axos = []
    # ax.crosshair = FinCrossHair(ax, color=cross_hair_color)
    # ax.hideButtons()
    # ax.overlay = partial(_ax_overlay, ax)
    # ax.set_visible = partial(_ax_set_visible, ax)
    # ax.decouple = partial(_ax_decouple, ax)
    # ax.disable_x_index = partial(_ax_disable_x_index, ax)
    # ax.reset = partial(_ax_reset, ax)
    # ax.invert_y = partial(_ax_invert_y, ax)
    # ax.expand = partial(_ax_expand, ax)
    # ax.prev_ax = prev_ax
    # ax.win_index = index
    if index % 2:
        viewbox.setBackgroundColor(theme["odd_plot_background"])
    viewbox.setParent(ax)
    return ax
