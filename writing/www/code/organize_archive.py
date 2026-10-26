"""Materialize paper/code/result associations and provenance without moving history."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'writing/www'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def link(dest,target):
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(os.path.relpath(target,dest.parent),target_is_directory=target.is_dir())


def main():
    snapshot=BASE/'code/project_snapshot'
    files=set((ROOT/'simulator_core').rglob('*.py'))
    for directory in ['v1_budget_20260922','v1_broad12h_20260925','v1_saving_threshold_20260924',
                      'v1_paper_comparison_20260922','kv_risk_feasibility_20260909/recovery_cost_v2']:
        files.update((ROOT/'research'/directory).glob('*.py'))
    files.update([ROOT/'run_paper_simulator_validation.py',ROOT/'tests/test_paper_simulator_validation.py'])
    files.update((BASE/'code').glob('*.py'))
    index={}
    for src in sorted(files):
        rel=src.relative_to(ROOT);dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);index[str(rel)]=sha(src)
    write(snapshot/'sha256.json',index)
    evidence=[]
    dirs=['v1_dynamic_actions_20260927','v1_completion_order_20260927','v1_fixed_order_20260927',
          'v1_candidate_audit_20260928','v1_candidate_transfer_20260928','v1_ratio_closed_loop_20260928',
          'v1_broad12h_20260925','v1_replica_extension_20260925']
    sources=[ROOT/'research'/d/'ANALYSIS_CN.md' for d in dirs]
    sources += [ROOT/'research/related_papers_review_20260923_CN.md',
        ROOT/'research/v1_paper_comparison_20260922_RESULTS_CN.md',
        ROOT/'research/v1_arrival_kv_quick_20260925/paper/PAPER_DRAFT_CN.md',
        ROOT/'research/v1_arrival_kv_quick_20260925/BASIC_RESULTS_CN.md',
        ROOT/'research/v1_replica_extension_20260925/RESULTS_INTERPRETATION_CN.md',
        ROOT/'research/sarathi_deep_read_20260928/READING_CN.md',
        ROOT/'docs/paper_simulator_validation.md',ROOT/'validation/paper_measurements/README.md']
    artifacts=['accepted_scheduler_baselines_20260917_fixed','v1_arrival_kv_quick_20260925',
        'qwen7b_full_metrics_20260926_r1','v1_mechanism_models_replicas_20260925_r1',
        'v1_ratio_closed_loop_20260928_r1','v1_candidate_transfer_20260928_r1']
    for directory in artifacts:
        for name in ['report.md','RESULTS_CN.md','summary.json','comparisons.csv',
                     'analysis/summary.json','analysis/audit.json','analysis/extra_validation.json']:
            sources.append(ROOT/'artifacts'/directory/name)
    for src in sources:
        if src.is_file():
            rel=src.relative_to(ROOT);dst=BASE/'evidence'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            evidence.append(dict(source=str(rel),snapshot=str(dst.relative_to(BASE)),sha256=sha(src)))
    write(BASE/'evidence/index.json',evidence)
    historical=BASE/'experiments/historical_20260917'
    shutil.copytree(ROOT/'artifacts/accepted_scheduler_baselines_20260917_fixed',historical,dirs_exist_ok=True)
    specs={
      'sarathi_2024':dict(policies=['sarathi'],native=['sarathi'],note='作者OSDI分支；共享V1规则适配',
        differences=[('scheduler/Algorithm 3','保留decode→partial prefill→FIFO新请求顺序。'),
        ('chunk/budget','作者固定chunk_size表示整轮budget；V1额外使用单请求K上限，本轮T1024/K256。不能称相同作者参数。'),
        ('memory','原版为完整prompt分配物理块；当前V1按实际推进增量分配。显著语义差异，须单列。'),
        ('execution','原kernel与TP/PP替换为CPU离散事件+合成时间；不声称原系统吞吐复现。')]),
      'ltr_2024':dict(policies=['ltr','ltr-oracle','mlfq'],native=[],note='作者代码已归档；现有结果仅规则/预测代理',
        differences=[('predictor','未运行作者OPT/ListMLE；trace可见分布p50/mean代替ranking分数，不能视作原模型精度。'),
        ('priority','保留(priority,score)和饥饿计数/quantum；原码aux_model_score采用反号排序，本地预测长度越小越优。'),
        ('memory/recovery','统一使用本地paged allocator与重算，未完整移植作者swap及评分推理开销。'),
        ('oracle','ltr-oracle使用真实输出长度，必须单列不可部署诊断。'),
        ('needed','原predictor见 https://huggingface.co/LLM-ltr/OPT-Predictors；需要对应真实prompt文本，现有长度trace不能直接喂给原模型。')]),
      's3_2023':dict(policies=['s3'],native=[],note='原文规则实现；作者原系统工件尚未定位',
        differences=[('predictor','trace可见长度统计替代DistilBERT，按10等宽桶向上取整。未复现作者训练及准确率。'),
        ('packing','保留预计KV降序first-fit以及不合适请求跳过。'),
        ('supervisor','超预留后翻倍，但host swap替换为丢弃/重算；迁移成本、后台更新尚未实现。')]),
      'vllm_2023':dict(policies=['fcfs','orca-max','orca-pow2','orca-oracle'],native=['native'],note='论文时代v0.2.0参考源码；主实验实际后端为0.9.2',
        differences=[('version','v0.2.0仅作SOSP时期源码参考，未确认是论文工件exact commit；当前V1 0.9.2是现代工程控制。'),
        ('allocator','旧通用实验为本地paged模型；新矩阵直接复用V1 allocator。均未执行PagedAttention GPU kernel。'),
        ('scope','单请求sampling；没有复现beam search、共享前缀、COW的吞吐与内存实验。')]),
      'orca_2022':dict(policies=['fcfs','orca-max','orca-pow2','orca-oracle'],native=[],note='原文调度/预留规则实现；原系统源码未定位',
        differences=[('iteration','FCFS iteration级调度与最大输出预留；未实现selective batching或多机流水。'),
        ('reservation','统一调度可见max_output_tokens=2048，避免从真实输出反推上限。'),
        ('oracle','Pow2/Oracle来自vLLM的Orca对照；读取真实输出长度，不是可部署Orca。')]),
      'past_future_2025':dict(policies=['past-future','conservative'],native=['past_future'],note='ASPLOS 2025；LightLLM HEAD不是已确认的论文版本',
        differences=[('history','最近1000个已完成长度，运行中按已生成长度条件采样；本地训练历史与测试输入分离。'),
        ('capacity','未来峰值改为物理block向上取整、V1保留1个null block。'),
        ('engine','保留FIFO遇unsafe停止；执行和恢复用V1，不是LightLLM原生路径。'),
        ('omissions','没有完整小batch多次采样优化、原版margin调参、跨框架工程及GPU计时。')]),
    }
    manifest=[]
    new=read_json(BASE/'experiments/native_simulator_20260928/summary.json')
    legacy=read_json(BASE/'experiments/legacy_rules_replay_20260928/summary.json')
    for key,spec in specs.items():
        folder=BASE/'baselines'/key
        upstream=folder/'code/upstream'
        provenance=dict(status=spec['note'],snapshot_date='2026-09-28',local_adapters_sha256=sha(ROOT/'simulator_core/paper_schedulers.py'))
        if upstream.is_dir():
            provenance.update(upstream_url=subprocess.check_output(['git','-C',str(upstream),'remote','get-url','origin'],text=True).strip(),
                upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
                upstream_modified=bool(subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()))
        write(folder/'code/provenance.json',provenance)
        shutil.copy2(ROOT/'simulator_core/paper_schedulers.py',folder/'code/paper_schedulers_snapshot.py')
        if spec['native']:
            shutil.copy2(ROOT/'research/v1_paper_comparison_20260922/scheduler.py',folder/'code/v1_rules_snapshot.py')
        link(folder/'code/shared_project_snapshot',snapshot)
        adapted=[dict(component=a,change=b) for a,b in spec['differences']]
        doc=[f'# {key} 复现与改动记录','',spec['note'],'','|部分|具体处理|','|---|---|']
        doc += [f'|{a}|{b}|' for a,b in spec['differences']]
        doc += ['','原文： [paper.pdf](paper/paper.pdf)。作者代码与commit见 [provenance.json](code/provenance.json)。',
            '本地代码快照保存于 code/；依赖源文件在 shared_project_snapshot/。快照用于审计，运行入口位于 ../../code/。',
            '结果见 results/。历史结果按原日期保留，本轮结果与历史结果分开，不复用旧数字。',
            '本轮只归档/读取作者仓库，未修改作者源代码；适配实现是本仓库的独立代码，不是一个可直接应用到作者仓库的补丁。']
        (folder/'ADAPTATION_CN.md').write_text('\n'.join(doc)+'\n')
        write(folder/'results/current_native_summary.json',[r for r in new if r['arm'] in spec['native']])
        write(folder/'results/current_legacy_summary.json',[r for r in legacy if r['identity']['policy'] in spec['policies']])
        link(folder/'results/native_raw',BASE/'experiments/native_simulator_20260928')
        link(folder/'results/legacy_raw',BASE/'experiments/legacy_rules_replay_20260928')
        link(folder/'results/historical_raw',historical)
        write(folder/'results/index.json',dict(legacy_policies=spec['policies'],native_arms=spec['native'],
            note='Empty summary means not included in this engine; do not compare latencies across engines. Raw links contain shared matrix; filter by listed policies.'))
        manifest.append(dict(id=key,paper=read_json(folder/'paper/source.json'),code=provenance,adaptations=adapted))
    write(BASE/'baselines/manifest.json',manifest)
    print('archived',len(manifest),'papers;',len(index),'local source files;',len(evidence),'evidence files')


def read_json(path):
    return json.loads(path.read_text())


if __name__=='__main__':
    main()
