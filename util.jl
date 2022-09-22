
"""Ensure filepath exists, creating directories if necessary."""
function ensure_filepath(filename::String)
    if !isdir(dirname(filename))
        mkpath(dirname(filename))
    end
end

"""Backup a file"""
function backup_file(filename)
    ensure_filepath(filename)
    filename_bak = "$(filename).bak"
    # filename_bak_bak = "$(filename_bak).bak"
    # if isfile(filename_bak)
    #     mv(filename_bak, filename_bak_bak, force=true)
    # end
    if isfile(filename)
        mv(filename, filename_bak, force=true)
    end
    return nothing
end
