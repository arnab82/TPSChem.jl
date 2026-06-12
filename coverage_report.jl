using Coverage, Printf

coverage = process_folder("src")
covered_total, total_total = get_summary(coverage)

println("=== Overall ===")
@printf("%.1f%%  (%d / %d executable lines)\n\n", 100.0*covered_total/total_total, covered_total, total_total)

println("=== Per-file (sorted by % covered) ===")
rows = Tuple{String,Int,Int,Float64}[]
for c in coverage
    cov = count(x -> x !== nothing && x > 0, c.coverage)
    tot = count(x -> x !== nothing, c.coverage)
    tot == 0 && continue
    f = replace(c.filename, pwd()*"/" => "")
    push!(rows, (f, cov, tot, 100.0*cov/tot))
end
sort!(rows, by=x->x[4])
for (f, cov, tot, pct) in rows
    bar = "" ^ round(Int, pct/5)
    @printf("  %5.1f%%  %4d/%-4d  %-20s  %s\n", pct, cov, tot, bar, f)
end
