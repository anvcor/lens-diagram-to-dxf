#!/usr/bin/env python3
"""CODE V COM 会话的薄封装（仅 Windows + 已装 CODE V + pywin32）。

    from cv_session import CodeV
    with CodeV(workdir=r'E:\\Download\\SIGMA85') as cv:
        cv.cmd('in "SIGMA85.seq"')
        efl = cv.eva('(EFL)')
        txt = cv.cmd('LIS')

要点：
  - 一定走 with：中途抛异常也会 StopCodeV()，否则后台留一个占着许可证的 codev.exe。
  - Command() 返回的是 CODE V 的文本输出；eva() 从 "(expr) = 值" 里把数抠出来。
  - CODE V 对出错不抛异常，只在文本里写 "Error"/"错误"；cmd(check=True) 时遇到就抛 CodeVError。
  - 路径含中文/空格时用双引号包住；工作目录用 SetStartingDirectory 定，之后可用相对路径。
"""
import re, sys

_NUM = re.compile(r'=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)')
_ERR = re.compile(r'\b(Error|ERROR)\b|错误')


class CodeVError(RuntimeError):
    pass


class CodeV:
    def __init__(self, workdir=None, progid='CodeV.Command', echo=False):
        self.workdir, self.progid, self.echo = workdir, progid, echo
        self.cv = None

    def __enter__(self):
        import win32com.client
        self.cv = win32com.client.Dispatch(self.progid)
        if self.workdir:
            self.cv.SetStartingDirectory(self.workdir)
        self.cv.StartCodeV()
        return self

    def __exit__(self, *exc):
        try:
            if self.cv is not None:
                self.cv.StopCodeV()
        finally:
            self.cv = None
        return False

    @property
    def version(self):
        return self.cv.GetCodeVVersion()

    def cmd(self, text, check=False):
        """发一条（或用 ; 连起来的几条）命令，返回文本输出。"""
        out = self.cv.Command(text)
        if self.echo:
            sys.stderr.write(f'CV> {text}\n{out}\n')
        if check and _ERR.search(out or ''):
            raise CodeVError(f'{text!r} → {out.strip()[:400]}')
        return out

    def eva(self, expr):
        """EVA 一个数值表达式，返回 float；取不到返回 None。expr 自带括号，如 '(EFL)'、'(RDY S3)'。"""
        out = self.cmd(f'EVA {expr}')
        m = _NUM.search(out or '')
        if not m:
            return None
        return float(m.group(1).replace('D', 'E').replace('d', 'e'))

    def run_seq(self, path, check=False):
        return self.cmd(f'in "{path}"', check=check)


if __name__ == '__main__':
    # 自检：python cv_session.py some.seq
    with CodeV() as cv:
        print('CODE V', cv.version)
        if len(sys.argv) > 1:
            cv.run_seq(sys.argv[1])
            print('EFL =', cv.eva('(EFL)'), ' 面数 =', cv.eva('(NUM S)'))
